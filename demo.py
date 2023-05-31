'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json

import opts
from tqdm import tqdm

import multiprocessing as mp
import threading
import glob


from tools.colormap import colormap


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

def main(args):
    args.masks = True
    args.batch_size == 1
    print("Inference only supports for batch size = 1")

    global transform
    transform = T.Compose([
    T.Resize(args.inf_res),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    global result_dict
    result_dict = mp.Manager().dict()
    frames = sorted(glob.glob(args.demo_path+'/*'))
    sub_processor(0, args, args.demo_exp, frames, save_path_prefix)

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

def sub_processor(pid, args, exp, frames, save_path_prefix):
    torch.cuda.set_device(pid)

    # model
    model, criterion, _ = build_model(args) 
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if pid == 0:
        print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
    	raise ValueError('Please specify the checkpoint for inference.')


    # start inference
    num_all_frames = 0 
    model.eval()

    sentence_features = []
    pseudo_sentence_features = []
    video_name = 'demo'
    # exp = meta[i]["exp"]
    # # exp = 'a dog is with its puppies on the cloth'
    # # TODO: temp
    # frames = meta[i]["frames"]
    # frames = [f'/home/mcg/ReferFormer/demo/frames_{fid}.jpg' for fid in range(1,2)]

    video_len = len(frames)
    # store images
    imgs = []
    for t in range(video_len):
        frame = frames[t]
        img_path = os.path.join(frame)
        img = Image.open(img_path).convert('RGB')
        origin_w, origin_h = img.size
        imgs.append(transform(img))  # list[img]

    imgs = torch.stack(imgs, dim=0).to(args.device) # [video_len, 3, h, w]
    img_h, img_w = imgs.shape[-2:]
    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
    target = {"size": size}

    with torch.no_grad():
        outputs = model([imgs], [exp], [target])

    pred_logits = outputs["pred_logits"][0]
    pred_boxes = outputs["pred_boxes"][0]
    pred_masks = outputs["pred_masks"][0]
    pred_ref_points = outputs["reference_points"][0]
    text_sentence_features = outputs['sentence_feature']
    if args.use_cycle:
        pseudo_text_sentence_features = outputs['pseudo_sentence_feature']
        # anchor = outputs['negative_anchor']
        sentence_features.append(text_sentence_features)
        pseudo_sentence_features.append(pseudo_text_sentence_features)
        # print(F.pairwise_distance(text_sentence_features, pseudo_text_sentence_features.squeeze(0), p=2))
    # print(anchor)
    # according to pred_logits, select the query index
    pred_scores = pred_logits.sigmoid()  # [t, q, k]
    pred_score = pred_scores
    pred_scores = pred_scores.mean(0)   # [q, k]
    max_scores, _ = pred_scores.max(-1)  # [q,]
    # print(max_scores)
    _, max_ind = max_scores.max(-1)     # [1,]
    max_inds = max_ind.repeat(video_len)
    pred_masks = pred_masks[range(video_len), max_inds, ...]  # [t, h, w]
    pred_masks = pred_masks.unsqueeze(0)
    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
    if args.save_prob:
        pred_masks = pred_masks.sigmoid().squeeze(0).detach().cpu().numpy()
    else:
        pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()
    if args.use_score:
        pred_score = pred_score[range(video_len), max_inds, 0].unsqueeze(-1).unsqueeze(-1)
        pred_masks *= (pred_score > 0.3).cpu().numpy() * pred_masks

    # store the video results
    all_pred_logits = pred_logits[range(video_len), max_inds].sigmoid().cpu().numpy()
    all_pred_boxes = pred_boxes[range(video_len), max_inds]
    all_pred_ref_points = pred_ref_points[range(video_len), max_inds]
    all_pred_masks = pred_masks

    save_path = os.path.join(save_path_prefix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for j in range(video_len):
        frame_name = frames[j]
        confidence = all_pred_logits[j]
        mask = all_pred_masks[j].astype(np.float32)
        save_file = os.path.join(save_path, f"{j}" + ".png")
        # print(save_file)
        if 'pair_logits' in outputs.keys() and args.use_cls:
            if outputs['pair_logits'].cpu().numpy() >= 0.5:
                print('This is a negative pair, disalignment degree:', outputs['pair_logits'].cpu().numpy().item())
            else:
                print('This is a positive pair, disalignment degree:', outputs['pair_logits'].cpu().numpy().item())
            mask *= 0 if outputs['pair_logits'].cpu().numpy() >= 0.5 else 1
        mask = Image.fromarray(mask * 255).convert('L')
        mask.save(save_file)
        print(f'Results saved to {save_path}')
    result_dict[str(pid)] = num_all_frames


# visuaize functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
        draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x-2, y-2, x+2, y+2), 
                            fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)
