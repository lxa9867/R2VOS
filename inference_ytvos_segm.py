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

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, split)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    if args.val_type:
        print(f'Warning: use robust rvos validation set, type: {args.val_type}')
        assert args.val_type in ['_random', '_color', '_old_color', '_object', '_action', '_dyn', '_1', '_3', '_5'], \
            f'Unknown val type for robust rvos, {args.val_type}'
        # load data
    root = Path(args.ytvos_path)  # data/ref-youtube-vos
    img_folder = os.path.join(root, split, f"JPEGImages{args.val_type}")
    meta_file = os.path.join(root, "meta_expressions", split, f"meta_expressions{args.val_type}.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    # for some reasons the competition's validation expressions dict contains both the validation (202) & 
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    if 'color' not in args.val_type:
        assert len(video_list) == 202 or len(video_list) == 202 + 305, 'error: incorrect number of validation videos'

    # create subprocess
    thread_num = args.ngpu
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data, 
                                                   save_path_prefix, save_visualize_path_prefix, 
                                                   img_folder, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" %(total_time))

def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
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
    # 1. For each video
    for vid, video in enumerate(video_list):
        metas = [] # list[dict], length is number of expressions

        expressions = data[video]["expressions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        vis = False
        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            # TODO: temp
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]
            if vis:
                os.makedirs('vis/tmp', exist_ok=True)

            video_len = len(frames)
            # store images
            imgs = []
            for t in range(video_len):
                frame = frames[t]
                img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                # img_path = frame
                img = Image.open(img_path).convert('RGB')
                if vis:
                    if t < 1:
                        img.save(f'vis/tmp/{t}.png')
                origin_w, origin_h = img.size
                imgs.append(transform(img))  # list[img]

            segm_logits = []
            segm_masks = []

            for segm_idx in range(video_len // args.segm_frame):
                if segm_idx != (video_len // args.segm_frame) - 1:
                    segm = imgs[segm_idx * args.segm_frame:(segm_idx+1) * args.segm_frame]
                else:
                    segm = imgs[segm_idx * args.segm_frame:]

                segm_len = len(segm)
                segm = torch.stack(segm, dim=0).to(args.device)  # [video_len, 3, h, w]
                img_h, img_w = segm.shape[-2:]
                size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                target = {"size": size}

                # start = time.time()
                with torch.no_grad():
                    outputs = model([segm], [exp], [target])
                # end = time.time()
                # print((end-start)/video_len)

                if vis:
                    os.system(f'mv vis/tmp vis/{vid}-{exp.replace(" ", "_")}')

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
                max_inds = max_ind.repeat(segm_len)
                pred_masks = pred_masks[range(segm_len), max_inds, ...]  # [t, h, w]
                pred_masks = pred_masks.unsqueeze(0)
                pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                if args.save_prob:
                    pred_masks = pred_masks.sigmoid().squeeze(0).detach().cpu().numpy()
                else:
                    pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()
                if args.use_score:
                    pred_score = pred_score[range(segm_len), max_inds, 0].unsqueeze(-1).unsqueeze(-1)
                    pred_masks *= (pred_score > 0.3).cpu().numpy() * pred_masks

                # store the video results
                all_pred_logits = pred_logits[range(segm_len), max_inds]
                all_pred_boxes = pred_boxes[range(segm_len), max_inds]
                all_pred_ref_points = pred_ref_points[range(segm_len), max_inds]
                all_pred_masks = pred_masks
                segm_logits.append(all_pred_logits.cpu().numpy())
                segm_masks.append(all_pred_masks)

            all_pred_logits = np.concatenate(segm_logits, axis=0)
            all_pred_masks = np.concatenate(segm_masks, axis=0)
            assert all_pred_masks.shape[0] == video_len
            # save binary image
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(video_len):
                frame_name = frames[j]
                mask = all_pred_masks[j].astype(np.float32)
                save_file = os.path.join(save_path, frame_name + ".png")
                if 'pair_logits' in outputs.keys() and args.use_cls:
                    mask *= 0 if outputs['pair_logits'].cpu().numpy() >= 0.5 else 1
                mask = Image.fromarray(mask * 255).convert('L')
                mask.save(save_file)

        with lock:
            progress.update(1)
    # bert_embed = torch.cat(sentence_features, dim=0)
    # np.save('output/bert_embed.npy', bert_embed.cpu().numpy())

    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


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
