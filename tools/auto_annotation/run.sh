conda activate ADMLOPS && \
cd $ADMLOPS && \
python3 ./tools/auto_annotation.py --input xxx/scalable_docker/items/xxx.json \
--type scalabel \
--config ./configs/yolox/yolox_l_8x8_300e_coco.py \
--checkpoint ./checkpoints/yolox/yolox_l_8x8_300e.pth
