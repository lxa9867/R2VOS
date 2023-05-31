from segmentation import *
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import cv2
import numpy as np
from einops import rearrange, repeat

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

import fvcore.nn.weight_init as weight_init

from .position_encoding import PositionEmbeddingSine1D

BN_MOMENTUM = 0.1

class S2EPrime(nn.Module):
    def __init__(self, feature_channels: List, conv_dim: int, mask_dim: int, dim_feedforward: int = 2048, norm=None,
                 return_query=False, stage_num=1):
        """
        Args:
            feature_channels: list of fpn feature channel numbers.
            conv_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            dim_feedforward: number of vision-language fusion module ffn channel numbers.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.feature_channels = feature_channels

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            # in_channels: 4x -> 32x
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)

            stage = idx + 1

            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        # vision-language cross-modal fusion
        self.text_pos = PositionEmbeddingSine1D(conv_dim, normalize=True)
        sr_ratios = [8, 4, 2, 1]

        self.stage_num = stage_num
        self.dyn_conv_list = []
        self.dyn_filter_list = []
        self.cross_attn_list = []
        self.pool_conv_list = []
        self.out_conv_list = []
        for i in range(self.stage_num):
            # init dyn conv
            output_norm = get_norm(norm, conv_dim)
            self.dyn_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            self.dyn_filter = nn.Linear(conv_dim, conv_dim, bias=True)
            weight_init.c2_xavier_fill(self.dyn_conv)
            weight_init.c2_xavier_fill(self.dyn_filter)

            # init text2video attn
            self.cross_attn = DualVisionLanguageBlock(conv_dim, dim_feedforward=dim_feedforward,
                                             nhead=8, sr_ratio=sr_ratios[-1])
            for p in self.cross_attn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            text_pool_size = 3
            self.pool = nn.AdaptiveAvgPool1d(text_pool_size)
            self.pool_conv = Conv2d(
                text_pool_size*conv_dim,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(self.pool_conv)
            # init out conv for stage != 0
            if i != 0:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.out_conv_list.append(output_conv)
            # add to list
            self.dyn_conv_list.append(self.dyn_conv)
            self.dyn_filter_list.append(self.dyn_filter)
            self.cross_attn_list.append(self.cross_attn)
            self.pool_conv_list.append(self.pool_conv)
        self.dyn_conv_list = nn.ModuleList(self.dyn_conv_list)
        self.dyn_filter_list = nn.ModuleList(self.dyn_filter_list)
        self.cross_attn_list = nn.ModuleList(self.cross_attn_list)
        self.pool_conv_list = nn.ModuleList(self.pool_conv_list)
        self.out_conv_list = nn.ModuleList(self.out_conv_list)
        print('use recurrent dual path cross attention')

    def forward_features(self, features, text_features, text_sentence_features, poses, memory, nf):
        # nf: num_frames
        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_features, text_masks = text_features.decompose()
        text_features = text_features.permute(1, 0, 2)

        for idx, (mem, f, pos) in enumerate(zip(memory[::-1], features[1:][::-1], poses[1:][::-1])):  # 32x -> 8x
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]

            _, x_mask = f.decompose()
            n, c, h, w = pos.shape
            b = n // nf
            t = nf

            # NOTE: here the (h, w) is the size for current fpn layer
            vision_features = lateral_conv(mem)  # [b*t, c, h, w]

            # upsample
            # TODO: only fuse in high-level and repeat
            if idx == 0:  # top layer
                for stage in range(self.stage_num):
                    if stage == 0:
                        vision_features = rearrange(vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
                    else:
                        vision_features = rearrange(y, '(b t) c h w -> (t h w) b c', b=b, t=t)
                    vision_pos = rearrange(pos, '(b t) c h w -> (t h w) b c', b=b, t=t)
                    vision_masks = rearrange(x_mask, '(b t) h w -> b (t h w)', b=b, t=t)
                    cur_fpn, text_features_ = self.cross_attn_list[stage](tgt=vision_features,
                                         memory=text_features,
                                         t=t, h=h, w=w,
                                         tgt_key_padding_mask=vision_masks,
                                         memory_key_padding_mask=text_masks,
                                         pos=text_pos,
                                         query_pos=vision_pos
                                         )  # [t*h*w, b, c]
                    # text_features [l, b, c]
                    cur_fpn = rearrange(cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                    # # repeat cls_token to compute filter, only apply once
                    # if text_sentence_features.shape[0] != b * t:
                    #     text_sentence_features = repeat(text_sentence_features, 'b c -> b t c', t=t)
                    #     text_sentence_features = rearrange(text_sentence_features, 'b t c -> (b t) c')
                    # filter = self.dyn_filter(text_sentence_features).unsqueeze(1)
                    # TODO: test only no fusion
                    text_features_ = repeat(text_features_, 'l b c -> l b t c', t=t)
                    text_features_ = rearrange(text_features_, 'l b t c -> (b t) c l')
                    text_features_ = rearrange(self.pool(text_features_), '(b t) c l -> (b t) l c', b=b, t=t)
                    filter = self.dyn_filter_list[stage](text_features_)
                    y = cur_fpn
                    y_ = self.dyn_conv_list[stage](cur_fpn)
                    y_ = torch.einsum('ichw,ijc -> ijchw', y_, filter)
                    y_ = rearrange(y_, 'i j c h w -> i (j c) h w')
                    y_ = self.pool_conv_list[stage](y_)
                    if stage == 0:
                        y = output_conv(y+y_)
                    else:
                        y = self.out_conv_list[stage-1](y+y_)
            else:
                # Following FPN implementation, we use nearest upsampling here
                cur_fpn = vision_features
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)

        # 4x level
        lateral_conv = self.lateral_convs[-1]
        output_conv = self.output_convs[-1]

        x, x_mask = features[0].decompose()
        pos = poses[0]
        n, c, h, w = pos.shape
        b = n // nf
        t = nf

        vision_features = lateral_conv(x)  # [b*t, c, h, w]
        cur_fpn = vision_features
        # Following FPN implementation, we use nearest upsampling here
        y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
        y = output_conv(y)
        return y

    def forward(self, features, text_features, text_sentence_features, pos, memory, nf):
        """The forward function receives the vision and language features,
            and outputs the mask features with the spatial stride of 4x.

        Args:
            features (list[NestedTensor]): backbone features (vision), length is number of FPN layers
                tensors: [b*t, ci, hi, wi], mask: [b*t, hi, wi]
            text_features (NestedTensor): text features (language)
                tensors: [b, length, c], mask: [b, length]
            pos (list[Tensor]): position encoding of vision features, length is number of FPN layers
                tensors: [b*t, c, hi, wi]
            memory (list[Tensor]): features from encoder output. from 8x -> 32x
            NOTE: the layer orders of both features and pos are res2 -> res5

        Returns:
            mask_features (Tensor): [b*t, mask_dim, h, w], with the spatial stride of 4x.
        """
        y = self.forward_features(features, text_features, text_sentence_features, pos, memory, nf)
        return self.mask_features(y)


# class EPrime2SPrime(nn.Module):
#
#
# class Cycle(nn.Module):
