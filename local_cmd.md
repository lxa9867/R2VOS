

# res50
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder --output_dir=output/refvos/res50_dual_memory_fpn_1 --resume=output/refvos/res50_dual_memory_fpn_1/checkpoint.pth --backbone resnet50 --ytvos_path /mnt/data/refvos --ngpu 1 --fpn_type 'dual+memory_fpn'
