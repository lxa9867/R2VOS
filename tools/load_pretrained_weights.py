import torch
from collections import OrderedDict

def pre_trained_model_to_finetune(checkpoint, args):
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
        # only delete the class_embed since the finetuned dataset has different num_classes
        num_layers = args.dec_layers + 1 if args.two_stage else args.dec_layers
        for l in range(num_layers):
            if "class_embed.{}.weight" in checkpoint.keys():
                del checkpoint["class_embed.{}.weight".format(l)]
                del checkpoint["class_embed.{}.bias".format(l)]
        # # determine backbone.0
        # flag = 0
        # for key in checkpoint.keys():
        #     if 'backbone' in key:
        #         flag = 1
        # if flag == 0:
        #     new_ckpt = OrderedDict()
        #     for k, v in checkpoint.items():
        #         if 'patch_embed' in k or 'attn.relative_position_' in k:
        #             continue
        #         new_ckpt['backbone.0.body.' + k] = v
        #     checkpoint = new_ckpt

    else:
        checkpoint = checkpoint['state_dict']
        new_ckpt = OrderedDict()
        for k, v in checkpoint.items():
            if 'patch_embed' in k:
                continue
            new_ckpt[k.replace('backbone', 'backbone.0.body')] = v
        checkpoint = new_ckpt
    return checkpoint
