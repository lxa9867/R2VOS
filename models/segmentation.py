"""
Segmentaion Part 
Modified from DETR (https://github.com/facebookresearch/detr)
"""
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

from sklearn.decomposition import PCA

BN_MOMENTUM = 0.1

def get_norm(norm, out_channels): # only support GN or LN
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

class Conv1d(torch.nn.Conv1d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        # if not torch.jit.is_scripting():
        #     if x.numel() == 0 and self.training:
        #         # https://github.com/pytorch/pytorch/issues/12013
        #         assert not isinstance(
        #             self.norm, torch.nn.SyncBatchNorm
        #         ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv1d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# FPN structure
class CrossModalFPNDecoder(nn.Module):
    def __init__(self, feature_channels: List, conv_dim: int, mask_dim: int, dim_feedforward: int = 2048, norm=None):
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
            stage = idx+1
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
        cross_attns = []
        for idx in range(len(feature_channels)): # res2 -> res5
            cross_attn = VisionLanguageBlock(conv_dim, dim_feedforward=dim_feedforward,
                                             nhead=8, sr_ratio=sr_ratios[idx])
            for p in cross_attn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            stage = int(idx + 1)
            self.add_module("cross_attn_{}".format(stage), cross_attn)
            cross_attns.append(cross_attn)
        # place cross-attn in top-down order (from low to high resolution)
        self.cross_attns = cross_attns[::-1]


    def forward_features(self, features, text_features, poses, memory, nf):
        # nf: num_frames
        text_pos = self.text_pos(text_features).permute(2, 0, 1)   # [length, batch_size, c]
        text_features, text_masks = text_features.decompose()
        text_features = text_features.permute(1, 0, 2)

        for idx, (mem, f, pos) in enumerate(zip(memory[::-1], features[1:][::-1], poses[1:][::-1])): # 32x -> 8x
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cross_attn = self.cross_attns[idx]

            _, x_mask = f.decompose()
            n, c, h, w = pos.shape
            b = n // nf
            t = nf

            # NOTE: here the (h, w) is the size for current fpn layer
            vision_features = lateral_conv(mem)  # [b*t, c, h, w]
            vision_features = rearrange(vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
            vision_pos = rearrange(pos, '(b t) c h w -> (t h w) b c', b=b, t=t)
            vision_masks = rearrange(x_mask, '(b t) h w -> b (t h w)', b=b, t=t)

            cur_fpn = cross_attn(tgt=vision_features,
                                 memory=text_features,
                                 t=t, h=h, w=w,
                                 tgt_key_padding_mask=vision_masks,
                                 memory_key_padding_mask=text_masks,
                                 pos=text_pos,
                                 query_pos=vision_pos
            ) # [t*h*w, b, c]
            cur_fpn = rearrange(cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            # upsample
            if idx == 0:  # top layer
                y = output_conv(cur_fpn)
            else:
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)

        # 4x level
        lateral_conv = self.lateral_convs[-1]
        output_conv = self.output_convs[-1]
        cross_attn = self.cross_attns[-1]

        x, x_mask = features[0].decompose()
        pos = poses[0]
        n, c, h, w = pos.shape
        b = n // nf
        t = nf

        vision_features = lateral_conv(x)  # [b*t, c, h, w]
        # TODO: disable low-level attn
        vision_features = rearrange(vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
        vision_pos = rearrange(pos, '(b t) c h w -> (t h w) b c', b=b, t=t)
        vision_masks = rearrange(x_mask, '(b t) h w -> b (t h w)', b=b, t=t)

        cur_fpn = cross_attn(tgt=vision_features,
                             memory=text_features,
                             t=t, h=h, w=w,
                             tgt_key_padding_mask=vision_masks,
                             memory_key_padding_mask=text_masks,
                             pos=text_pos,
                             query_pos=vision_pos
        )  # [t*h*w, b, c]
        cur_fpn = rearrange(cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
        # cur_fpn = vision_features
        # Following FPN implementation, we use nearest upsampling here
        y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
        y = output_conv(y)
        return y   # [b*t, c, h, w], the spatial stride is 4x

    def forward(self, features, text_features, pos, memory, nf):
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
        y = self.forward_features(features, text_features, pos, memory, nf)
        return self.mask_features(y)

class DualCrossModalFPNDecoder(nn.Module):
    def __init__(self, feature_channels: List, conv_dim: int, mask_dim: int, dim_feedforward: int = 2048, norm=None,
                 use_cycle=False, add_negative=False):
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

        # init dyn conv
        dyn_norm = get_norm(norm, conv_dim)
        self.dyn_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
            norm=dyn_norm,
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
        pool_norm = get_norm(norm, conv_dim)
        self.pool = nn.AdaptiveAvgPool1d(text_pool_size)
        self.pool_conv = Conv2d(
            text_pool_size*conv_dim,
            conv_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
            norm=pool_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(self.pool_conv)
        if use_cycle:
            d_modal = 256
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_modal, nhead=8)
            decoder_norm = nn.LayerNorm(d_modal)
            self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3, norm=decoder_norm)
            self.text_query = nn.Embedding(1, 256)
        self.use_cycle = use_cycle
        self.add_negative = add_negative
        print('use dual path cross attention')

    def forward_features(self, features, text_features, poses, memory, nf, neg_memory=None, neg_text_features=None):
        # nf: num_frames
        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_features, text_masks = text_features.decompose()
        text_features = text_features.permute(1, 0, 2)
        out_feats = []

        if self.add_negative:
            neg_text_pos = self.text_pos(neg_text_features).permute(2, 0, 1)  # [length, batch_size, c]
            neg_text_features, neg_text_masks = neg_text_features.decompose()
            neg_text_features = neg_text_features.permute(1, 0, 2)


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
            # TODO: only fuse in high-level
            if idx == 0:  # top layer
                vision_features = rearrange(vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
                vision_pos = rearrange(pos, '(b t) c h w -> (t h w) b c', b=b, t=t)
                vision_masks = rearrange(x_mask, '(b t) h w -> b (t h w)', b=b, t=t)
                text_features_before_fusion = text_features
                vision_features_before_fusion = vision_features
                cur_fpn, text_features = self.cross_attn(tgt=vision_features,
                                     memory=text_features,
                                     t=t, h=h, w=w,
                                     tgt_key_padding_mask=vision_masks,
                                     memory_key_padding_mask=text_masks,
                                     pos=text_pos,
                                     query_pos=vision_pos
                                     )  # [t*h*w, b, c]
                # text_features [l, b, c]
                cur_fpn = rearrange(cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                # TODO: test only no fusion
                text_features = repeat(text_features, 'l b c -> l b t c', t=t)
                text_features = rearrange(text_features, 'l b t c -> (b t) c l')
                text_features = rearrange(self.pool(text_features), '(b t) c l -> (b t) l c', b=b, t=t)
                filter = self.dyn_filter(text_features)
                y = cur_fpn
                y_ = self.dyn_conv(cur_fpn)
                y_ = torch.einsum('ichw,ijc -> ijchw', y_, filter)
                # vis_dyn = True
                # if vis_dyn:
                #     # TODO: Vis dyn conv
                #     for i in range(1):
                #         frame = cv2.imread(f'vis/tmp/{i}.png')
                #         frame_hw = frame.shape[:2]
                #         for j in range(3):
                #             vis_feat = torch.nn.functional.relu(torch.sum(y_[i, j], dim=0))
                #             # vis_feat = y_[i, j, 100].abs()
                #             vis_feat /= vis_feat.max()
                #             vis_feat *= 255
                #             vis_feat = vis_feat.detach().cpu().numpy().astype(np.uint8)
                #             vis_feat = cv2.resize(vis_feat, frame_hw[::-1], interpolation=cv2.INTER_NEAREST)
                #             vis_feat = cv2.applyColorMap(vis_feat, cv2.COLORMAP_HOT)
                #             vis_feat = cv2.addWeighted(frame, 0, vis_feat, 1, 0.0)
                #             cv2.imwrite(f'vis/tmp/frame_{i}_filter_{j}.png', vis_feat)
                y_ = rearrange(y_, 'i j c h w -> i (j c) h w')
                y_ = self.pool_conv(y_)
                y = output_conv(y+y_)
                # video captioning
                if self.use_cycle:
                    text_query = self.text_query.weight.repeat(1, b, 1)  # 1xBxC

                    if self.add_negative:
                        neg_vision_features = lateral_conv(neg_memory)
                        neg_vision_features = rearrange(neg_vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
                        neg_cur_fpn, neg_text_features = self.cross_attn(tgt=neg_vision_features,
                                                                 memory=neg_text_features,
                                                                 t=t, h=h, w=w,
                                                                 tgt_key_padding_mask=vision_masks,
                                                                 memory_key_padding_mask=neg_text_masks,
                                                                 pos=neg_text_pos,
                                                                 query_pos=vision_pos
                                                                 )  # [t*h*w, b, c]
                        # text_features [l, b, c]
                        neg_cur_fpn = rearrange(neg_cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                        # TODO: test only no fusion
                        neg_text_features = repeat(neg_text_features, 'l b c -> l b t c', t=t)
                        neg_text_features = rearrange(neg_text_features, 'l b t c -> (b t) c l')
                        neg_text_features = rearrange(self.pool(neg_text_features), '(b t) c l -> (b t) l c', b=b, t=t)
                        neg_filter = self.dyn_filter(neg_text_features)
                        neg_y = neg_cur_fpn
                        neg_y_ = self.dyn_conv(neg_cur_fpn)
                        neg_y_ = torch.einsum('ichw,ijc -> ijchw', neg_y_, neg_filter)
                        neg_y_ = rearrange(neg_y_, 'i j c h w -> i (j c) h w')
                        neg_y_ = self.pool_conv(neg_y_)
                        neg_y = output_conv(neg_y + neg_y_)
                        neg_text_aware_video_feature = vision_pos + rearrange(neg_y, '(b t) c h w -> (t h w) b c', b=b, t=t)
                        neg_pseudo_sentence_feature = self.text_decoder(text_query, neg_text_aware_video_feature,
                                                                    None, vision_masks)

                    text_aware_video_feature = vision_pos + rearrange(y, '(b t) c h w -> (t h w) b c', b=b, t=t)
                    pseudo_sentence_feature = self.text_decoder(text_query, text_aware_video_feature,
                                                                None, vision_masks[0].unsqueeze(0))
            else:
                # Following FPN implementation, we use nearest upsampling here
                cur_fpn = vision_features
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if idx < 4:
                out_feats.append(y)

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
        if self.use_cycle:
            if self.add_negative:
                return y, out_feats, pseudo_sentence_feature, neg_pseudo_sentence_feature
            return y, out_feats, pseudo_sentence_feature
        return y, out_feats

    def forward(self, features, text_features, text_sentence_features, pos, memory, nf, neg_memory=None,
                neg_text_features=None):
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
        if self.use_cycle:
            if self.add_negative:
                y, out_feat, pseudo_sentence_feature, neg_pseudo_sentence_feature = self.forward_features(features, text_features, pos, memory, nf,
                                                                             neg_memory, neg_text_features)
                return self.mask_features(y), out_feat, pseudo_sentence_feature, neg_pseudo_sentence_feature
            else:
                y, out_feat, pseudo_sentence_feature = self.forward_features(features, text_features, pos, memory, nf)
                return self.mask_features(y), out_feat, pseudo_sentence_feature
        y, out_feat = self.forward_features(features, text_features, pos, memory, nf)
        return self.mask_features(y), out_feat

class RecurrentDualCrossModalFPNDecoder(nn.Module):
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
            dyn_norm = get_norm(norm, conv_dim)
            self.dyn_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=dyn_norm,
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
            pool_norm = get_norm(norm, conv_dim)
            self.pool = nn.AdaptiveAvgPool1d(text_pool_size)
            self.pool_conv = Conv2d(
                text_pool_size*conv_dim,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=pool_norm,
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
                    # vis_dyn = True
                    # if vis_dyn:
                    #     # TODO: Vis dyn conv
                    #     for i in range(t):
                    #         frame = cv2.imread(f'vis/tmp/{i}.png')
                    #         frame_hw = frame.shape[:2]
                    #         for j in range(3):
                    #             vis_feat = torch.nn.functional.relu(-torch.sum(y_[i, j], dim=0))
                    #             # vis_feat = y_[i, j, 100].abs()
                    #             vis_feat /= vis_feat.max()
                    #             vis_feat *= 255
                    #             vis_feat = vis_feat.detach().cpu().numpy().astype(np.uint8)
                    #             vis_feat = cv2.resize(vis_feat, frame_hw[::-1], interpolation=cv2.INTER_NEAREST)
                    #             vis_feat = cv2.applyColorMap(vis_feat, cv2.COLORMAP_HOT)
                    #             vis_feat = cv2.addWeighted(frame, 0, vis_feat, 1, 0.0)
                    #             cv2.imwrite(f'vis/tmp/frame_{i}_filter_{j}_stage_{stage}.png', vis_feat)
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

class DualMultiScaleCrossModalFPNDecoder(nn.Module):
    def __init__(self, feature_channels: List, conv_dim: int, mask_dim: int, dim_feedforward: int = 2048, norm=None, text_pool_size=3):
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
        dyn_convs = []
        dyn_filters = []
        pool_convs = []
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
            # for AMM
            dyn_norm = get_norm(norm, conv_dim)
            dyn_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=dyn_norm,
                activation=F.relu,
            )
            dyn_filter = nn.Linear(conv_dim, conv_dim, bias=True)
            pool_norm = get_norm(norm, conv_dim)
            pool_conv = Conv2d(
                text_pool_size * conv_dim,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=pool_norm,
                activation=F.relu,
            )

            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            weight_init.c2_xavier_fill(dyn_conv)
            weight_init.c2_xavier_fill(dyn_filter)
            weight_init.c2_xavier_fill(pool_conv)

            stage = idx+1

            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)
            self.add_module("dyn_conv_{}".format(stage), dyn_conv)
            self.add_module("dyn_filter_{}".format(stage), dyn_filter)
            self.add_module("pool_conv_{}".format(stage), pool_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
            dyn_convs.append(dyn_conv)
            dyn_filters.append(dyn_filter)
            pool_convs.append(pool_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.dyn_convs = dyn_convs[::-1]
        self.dyn_filters = dyn_filters[::-1]
        self.pool_convs = pool_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        # for CMM dynamic filter
        self.pool = nn.AdaptiveAvgPool1d(text_pool_size)
        # vision-language cross-modal fusion
        self.text_pos = PositionEmbeddingSine1D(conv_dim, normalize=True)
        sr_ratios = [8, 4, 2, 1]
        cross_attns = []
        for idx in range(len(feature_channels)): # res2 -> res5
            cross_attn = MSDualVisionLanguageBlock(conv_dim, dim_feedforward=dim_feedforward,
                                             nhead=8, sr_ratio=sr_ratios[idx])
            for p in cross_attn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            stage = int(idx + 1)
            self.add_module("cross_attn_{}".format(stage), cross_attn)
            cross_attns.append(cross_attn)
        # place cross-attn in top-down order (from low to high resolution)
        self.cross_attns = cross_attns[::-1]

    def forward_features(self, features, text_features, text_sentence_features, poses, memory, nf):
        # nf: num_frames
        text_pos = self.text_pos(text_features).permute(2, 0, 1)   # [length, batch_size, c]
        text_features, text_masks = text_features.decompose()
        text_features = text_features.permute(1, 0, 2)

        for idx, (mem, f, pos) in enumerate(zip(memory[::-1], features[1:][::-1], poses[1:][::-1])): # 32x -> 8x
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cross_attn = self.cross_attns[idx]
            dyn_filter = self.dyn_filters[idx]
            dyn_conv = self.dyn_convs[idx]
            pool_conv = self.pool_convs[idx]

            _, x_mask = f.decompose()
            n, c, h, w = pos.shape
            b = n // nf
            t = nf

            # NOTE: here the (h, w) is the size for current fpn layer
            vision_features = lateral_conv(mem)  # [b*t, c, h, w]
            vision_features = rearrange(vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
            vision_pos = rearrange(pos, '(b t) c h w -> (t h w) b c', b=b, t=t)
            vision_masks = rearrange(x_mask, '(b t) h w -> b (t h w)', b=b, t=t)

            cur_fpn, video_aware_text_features = cross_attn(tgt=vision_features,
                                 memory=text_features,
                                 t=t, h=h, w=w,
                                 tgt_key_padding_mask=vision_masks,
                                 memory_key_padding_mask=text_masks,
                                 pos=text_pos,
                                 query_pos=vision_pos
            ) # [t*h*w, b, c]
            cur_fpn = rearrange(cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            video_aware_text_features = repeat(video_aware_text_features, 'l b c -> l b t c', t=t)
            video_aware_text_features = rearrange(video_aware_text_features, 'l b t c -> (b t) c l')
            video_aware_text_features = rearrange(self.pool(video_aware_text_features), '(b t) c l -> (b t) l c', b=b, t=t)
            filter = dyn_filter(video_aware_text_features)

            # upsample
            if idx == 0:  # top layer
                y = cur_fpn
                y_ = dyn_conv(cur_fpn)
                y_ = torch.einsum('ichw,ijc -> ijchw', y_, filter)
                y_ = rearrange(y_, 'i j c h w -> i (j c) h w')
                y_ = pool_conv(y_)
                y = output_conv(y + y_)
            else:
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y_ = dyn_conv(y)
                y_ = torch.einsum('ichw,ijc -> ijchw', y_, filter)
                y_ = rearrange(y_, 'i j c h w -> i (j c) h w')
                y_ = pool_conv(y_)
                y = output_conv(y + y_)

        # 4x level
        lateral_conv = self.lateral_convs[-1]
        output_conv = self.output_convs[-1]
        cross_attn = self.cross_attns[-1]
        dyn_filter = self.dyn_filters[-1]
        dyn_conv = self.dyn_convs[-1]
        pool_conv = self.pool_convs[-1]

        x, x_mask = features[0].decompose()
        pos = poses[0]
        n, c, h, w = pos.shape
        b = n // nf
        t = nf

        vision_features = lateral_conv(x)  # [b*t, c, h, w]
        # TODO: disable low-level attn
        vision_features = rearrange(vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
        vision_pos = rearrange(pos, '(b t) c h w -> (t h w) b c', b=b, t=t)
        vision_masks = rearrange(x_mask, '(b t) h w -> b (t h w)', b=b, t=t)

        cur_fpn, video_aware_text_features = cross_attn(tgt=vision_features,
                             memory=text_features,
                             t=t, h=h, w=w,
                             tgt_key_padding_mask=vision_masks,
                             memory_key_padding_mask=text_masks,
                             pos=text_pos,
                             query_pos=vision_pos
        )  # [t*h*w, b, c]
        cur_fpn = rearrange(cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
        video_aware_text_features = repeat(video_aware_text_features, 'l b c -> l b t c', t=t)
        video_aware_text_features = rearrange(video_aware_text_features, 'l b t c -> (b t) c l')
        video_aware_text_features = rearrange(self.pool(video_aware_text_features), '(b t) c l -> (b t) l c', b=b, t=t)
        filter = dyn_filter(video_aware_text_features)
        # cur_fpn = vision_features
        # Following FPN implementation, we use nearest upsampling here
        y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
        y_ = dyn_conv(y)
        y_ = torch.einsum('ichw,ijc -> ijchw', y_, filter)
        y_ = rearrange(y_, 'i j c h w -> i (j c) h w')
        y_ = pool_conv(y_)
        y = output_conv(y + y_)
        return y   # [b*t, c, h, w], the spatial stride is 4x

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

class VisionLanguageBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sr_ratio=1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # for downsample
        self.sr_ratio = sr_ratio

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, t, h, w,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        b = tgt.size(1)
        # self attn
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.sr_ratio > 1: # downsample
            q = rearrange(q, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            k = rearrange(k, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            v = rearrange(tgt, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            # downsample
            new_h = int(h * 1./self.sr_ratio)
            new_w = int(w * 1./self.sr_ratio)
            size = (new_h, new_w)
            q = F.interpolate(q, size=size, mode='nearest')
            k = F.interpolate(k, size=size, mode='nearest')
            v = F.interpolate(v, size=size, mode='nearest')
            # shape for transformer
            q = rearrange(q, '(b t) c h w -> (t h w) b c', t=t)
            k = rearrange(k, '(b t) c h w -> (t h w) b c', t=t)
            v = rearrange(v, '(b t) c h w -> (t h w) b c', t=t)
            # downsample mask
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b*t, h, w)
            tgt_key_padding_mask = F.interpolate(tgt_key_padding_mask[None].float(), size=(new_h, new_w), mode='nearest').bool()[0] 
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b, t, new_h, new_w).flatten(1)
        else:
            v = tgt
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0] # [H*W, B*T, C]
        if self.sr_ratio > 1:
            tgt2 = rearrange(tgt2, '(t h w) b c -> (b t) c h w', t=t, h=new_h, w=new_w)
            size = (h, w)  # recover to origin size
            tgt2 = F.interpolate(tgt2, size=size, mode='bilinear', align_corners=False) # [B*T, C, H, W]
            tgt2 = rearrange(tgt2, '(b t) c h w -> (t h w) b c', t=t)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attn
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, t, h, w,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        b = tgt.size(1)
        # self attn
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        if self.sr_ratio > 1: # downsample
            q = rearrange(q, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            k = rearrange(k, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            v = rearrange(tgt, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            # downsample
            new_h = int(h * 1./self.sr_ratio)
            new_w = int(w * 1./self.sr_ratio)
            size = (new_h, new_w)
            q = F.interpolate(q, size=size, mode='nearest')
            k = F.interpolate(k, size=size, mode='nearest')
            v = F.interpolate(v, size=size, mode='nearest')
            # shape for transformer
            q = rearrange(q, '(b t) c h w -> (t h w) b c', t=t)
            k = rearrange(k, '(b t) c h w -> (t h w) b c', t=t)
            v = rearrange(v, '(b t) c h w -> (t h w) b c', t=t)
            # downsample mask
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b*t, h, w)
            tgt_key_padding_mask = F.interpolate(tgt_key_padding_mask[None].float(), size=(new_h, new_w), mode='nearest').bool()[0] 
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b, t, new_h, new_w).flatten(1)
        else:
            v = tgt2
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0]  # [T*H*W, B, C]
        if self.sr_ratio > 1:
            tgt2 = rearrange(tgt2, '(t h w) b c -> (b t) c h w', t=t, h=new_h, w=new_w)
            size = (h, w)  # recover to origin size
            tgt2 = F.interpolate(tgt2, size=size, mode='bilinear', align_corners=False) # [B*T, C, H, W]
            tgt2 = rearrange(tgt2, '(b t) c h w -> (t h w) b c', t=t)
        tgt = tgt + self.dropout1(tgt2)

        # cross attn
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, t, h, w,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, t, h, w,
                                    tgt_key_padding_mask, memory_key_padding_mask, 
                                    pos, query_pos)
        return self.forward_post(tgt, memory, t, h, w,
                                 tgt_key_padding_mask, memory_key_padding_mask, 
                                 pos, query_pos)


class DualVisionLanguageBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sr_ratio=1):
        super().__init__()
        self.v2v_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.l2v_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.v2l_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.proj_l_q = nn.Linear(d_model, d_model)
        self.proj_v_kv = nn.Linear(d_model, 2*d_model)
        self.d_modal = d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_v2l_1 = nn.Dropout(dropout)
        self.dropout_v2l_2 = nn.Dropout(dropout)
        self.dropout_v2l_3 = nn.Dropout(dropout)
        self.linear_v2l_1 = nn.Linear(d_model, dim_feedforward)
        self.linear_v2l_2 = nn.Linear(dim_feedforward, d_model)
        self.norm_v2l_1 = nn.LayerNorm(d_model)
        self.norm_v2l_2 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.activation_v2l = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # for downsample
        self.sr_ratio = sr_ratio

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, t, h, w,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        b = tgt.size(1)
        # v2v attn
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        tgt2 = self.v2v_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0]  # [H*W, B*T, C]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # l2v attn
        tgt2 = self.l2v_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]

        # v2l attn
        l_q = self.proj_l_q(memory)
        v_kv = self.proj_v_kv(tgt)
        v_k, v_v = torch.split(v_kv, [self.d_modal, self.d_modal], dim=-1)

        memory2 = self.v2l_attn(query=self.with_pos_embed(l_q, pos),
                                key=self.with_pos_embed(v_k, query_pos),
                                value=v_v, attn_mask=None,
                                key_padding_mask=tgt_key_padding_mask)[0]

        # l2v add & norm & ffn
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # v2l add & norm & ffn
        memory = memory + self.dropout_v2l_1(memory2)
        memory = self.norm_v2l_1(memory)

        memory2 = self.linear_v2l_2(self.dropout_v2l_2(self.activation_v2l(self.linear_v2l_1(memory))))
        memory = memory + self.dropout_v2l_3(memory2)
        memory = self.norm_v2l_2(memory)

        return tgt, memory

    def forward(self, tgt, memory, t, h, w,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            raise NotImplementedError
        return self.forward_post(tgt, memory, t, h, w,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


class MSDualVisionLanguageBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sr_ratio=1):
        super().__init__()
        self.v2v_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.l2v_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.v2l_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.proj_l_q = nn.Linear(d_model, d_model)
        self.proj_v_kv = nn.Linear(d_model, 2 * d_model)
        self.d_modal = d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_v2l_1 = nn.Dropout(dropout)
        self.dropout_v2l_2 = nn.Dropout(dropout)
        self.dropout_v2l_3 = nn.Dropout(dropout)
        self.linear_v2l_1 = nn.Linear(d_model, dim_feedforward)
        self.linear_v2l_2 = nn.Linear(dim_feedforward, d_model)
        self.norm_v2l_1 = nn.LayerNorm(d_model)
        self.norm_v2l_2 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.activation_v2l = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # for downsample
        self.sr_ratio = sr_ratio

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, t, h, w,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        b = tgt.size(1)
        # v2v attn
        q = k = self.with_pos_embed(tgt, query_pos)
        ori_tgt_key_padding_mask = tgt_key_padding_mask
        if self.sr_ratio > 1:  # downsample
            q = rearrange(q, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            k = rearrange(k, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            v = rearrange(tgt, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            # downsample
            new_h = int(h * 1. / self.sr_ratio)
            new_w = int(w * 1. / self.sr_ratio)
            size = (new_h, new_w)
            q = F.interpolate(q, size=size, mode='nearest')
            k = F.interpolate(k, size=size, mode='nearest')
            v = F.interpolate(v, size=size, mode='nearest')
            # shape for transformer
            q = rearrange(q, '(b t) c h w -> (t h w) b c', t=t)
            k = rearrange(k, '(b t) c h w -> (t h w) b c', t=t)
            v = rearrange(v, '(b t) c h w -> (t h w) b c', t=t)
            # downsample mask
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b * t, h, w)
            tgt_key_padding_mask = \
            F.interpolate(tgt_key_padding_mask[None].float(), size=(new_h, new_w), mode='nearest').bool()[0]
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b, t, new_h, new_w).flatten(1)
        else:
            v = tgt
        tgt2 = self.v2v_attn(q, k, value=v, attn_mask=None,
                             key_padding_mask=tgt_key_padding_mask)[0]  # [H*W, B*T, C]

        if self.sr_ratio > 1:
            tgt2 = rearrange(tgt2, '(t h w) b c -> (b t) c h w', t=t, h=new_h, w=new_w)
            size = (h, w)  # recover to origin size
            tgt2 = F.interpolate(tgt2, size=size, mode='bilinear', align_corners=False)  # [B*T, C, H, W]
            tgt2 = rearrange(tgt2, '(b t) c h w -> (t h w) b c', t=t)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # l2v attn
        tgt2 = self.l2v_attn(query=self.with_pos_embed(tgt, query_pos),
                             key=self.with_pos_embed(memory, pos),
                             value=memory, attn_mask=None,
                             key_padding_mask=memory_key_padding_mask)[0]

        # v2l attn
        l_q = self.proj_l_q(memory)
        v_kv = self.proj_v_kv(tgt)
        v_k, v_v = torch.split(v_kv, [self.d_modal, self.d_modal], dim=-1)

        memory2 = self.v2l_attn(query=self.with_pos_embed(l_q, pos),
                                key=self.with_pos_embed(v_k, query_pos),
                                value=v_v, attn_mask=None,
                                key_padding_mask=ori_tgt_key_padding_mask)[0]

        # l2v add & norm & ffn
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # v2l add & norm & ffn
        memory = memory + self.dropout_v2l_1(memory2)
        memory = self.norm_v2l_1(memory)

        memory2 = self.linear_v2l_2(self.dropout_v2l_2(self.activation_v2l(self.linear_v2l_1(memory))))
        memory = memory + self.dropout_v2l_3(memory2)
        memory = self.norm_v2l_2(memory)

        return tgt, memory

    def forward(self, tgt, memory, t, h, w,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            raise NotImplementedError
        return self.forward_post(tgt, memory, t, h, w,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


