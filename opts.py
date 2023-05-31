import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('ReferFormer training and inference scripts.', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--lr_text_encoder', default=1e-5, type=float)
    parser.add_argument('--lr_text_encoder_names', default=['text_encoder'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    parser.add_argument('--lr_multi', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=[8, 10], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    # load the pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help="Path to the pretrained model.") 

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')  # NOTE: must be false

    # * Backbone
    # ["resnet50", "resnet101", "swin_t_p4w7", "swin_s_p4w7", "swin_b_p4w7", "swin_l_p4w7"]
    # ["video_swin_t_p4w7", "video_swin_s_p4w7", "video_swin_b_p4w7"]
    parser.add_argument('--backbone', default='resnet50', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_pretrained', default=None, type=str, 
                        help="if use swin backbone and train from scratch, the path to the pretrained weights")
    parser.add_argument('--use_checkpoint', action='store_true', help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--dilation', action='store_true',  # DC5
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, 
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots, all frames share the same queries") 
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    # for text
    parser.add_argument('--freeze_text_encoder', action='store_true') # default: False

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_dim', default=256, type=int, 
                        help="Size of the mask embeddings (dimension of the dynamic mask conv)")
    parser.add_argument('--controller_layers', default=3, type=int, 
                        help="Dynamic conv layer number")
    parser.add_argument('--dynamic_mask_channels', default=8, type=int, 
                        help="Dynamic conv final channel number")
    parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false',
                        help="Disables relative coordinates")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_mask', default=2, type=float,
                        help="mask coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=5, type=float,
                        help="mask coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    # ['ytvos', 'davis', 'a2d', 'jhmdb', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasets for pretraining
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name') 
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--ytvos_path', type=str, default='data/ref-youtube-vos')
    parser.add_argument('--davis_path', type=str, default='data/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='data/a2d_sentences')
    parser.add_argument('--jhmdb_path', type=str, default='/mnt/data/jhmdb')
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=640, type=int, help="max size for the frame")
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # test setting
    parser.add_argument('--threshold', default=0.5, type=float) # binary threshold for mask
    parser.add_argument('--ngpu', default=8, type=int, help='gpu number when inference for ref-ytvos and ref-davis')
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'test'])
    parser.add_argument('--visualize', action='store_true', help='whether visualize the masks during inference')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # additional parameters
    parser.add_argument('--fpn_type', default='dual', help='fpn type can be dual, dyn, and default')
    parser.add_argument('--val_type', default='', help='validation dataset type for youtube rvos')
    parser.add_argument('--as_vos', default=False, help='as semi-supervised vos')
    parser.add_argument('--query_feat_dim', default=2048, help='feat_dim of 1/32 visual feature map')
    parser.add_argument('--inf_res', default=360, type=int, help='inference size')
    parser.add_argument('--text_enc_type', default='distilroberta-base', help='fpn type can be dual, dyn, and default')
    parser.add_argument('--use_cycle', action='store_true', help='use cycle consistency')
    parser.add_argument('--add_negative', action='store_true', help='add negative sample on gpu 0 for triplet loss')
    parser.add_argument('--only_cycle', action='store_true', help='only train cycle consistency part model')
    parser.add_argument('--cycle_loss_dist_coef', default=1, type=float)
    parser.add_argument('--cycle_loss_angle_coef', default=1, type=float)
    parser.add_argument('--cycle_loss_mse_coef', default=0.0, type=float)
    parser.add_argument('--cycle_loss_cls_coef', default=1, type=float)
    parser.add_argument('--fg_contra_loss_coef', default=1, type=float)
    parser.add_argument('--VQ_loss_coef', default=0.5, type=float)
    parser.add_argument('--cycle_loss_contrastive_coef', default=0.1, type=float)
    parser.add_argument('--loc_loss_coef', default=3, type=float)
    parser.add_argument('--lr_anchor_names', default=['negative_anchor'], type=str, nargs='+')
    parser.add_argument('--lr_anchor_mult', default=0.1, type=float)
    parser.add_argument('--contra_margin', default=0.5, type=float)
    parser.add_argument('--is_eval', action='store_true', help='use in eval')
    parser.add_argument('--neg_cls', action='store_true', help='add classifier to classify neg samples')
    parser.add_argument('--bert_cycle', action='store_true', help='use 768 dim output from bert as pos gt')
    parser.add_argument('--mix_query', action='store_true', help='mix pseudo-text and text query to deformable trans')
    parser.add_argument('--quantitize_query', action='store_true', help='quantitize text query')
    parser.add_argument('--use_fg_contra', action='store_true', help='use fg contra loss')
    parser.add_argument('--freeze_quantitizer', action='store_true', help='freeze quantitizer')
    parser.add_argument('--pseudo_label_path', default='', help='pseudo label path')
    parser.add_argument('--use_cls', action='store_true', help='use neg cls to filter out negative videos')
    parser.add_argument('--use_score', action='store_true', help='use score to filter out negative videos')
    parser.add_argument('--save_prob', action='store_true', help='save prob map')
    parser.add_argument('--segm_frame', default=5, type=int)
    parser.add_argument('--demo_exp', default='a big track on the road', help='demo exp')
    parser.add_argument('--demo_path', default='demo/demo_examples', help='demo frames folder path')
    return parser


