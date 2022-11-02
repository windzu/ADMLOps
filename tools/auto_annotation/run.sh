conda activate mmdet && \
cd $ADMLOPS_PATH/mmdetection_extension/tools/auto_annotation && \
python main.py \
--input /home/wind/Projects/wind_activate/scalable_docker/items/local_weitang_image_list1.json \
--type scalabel \
--config $ADMLOPS_PATH/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py \
--checkpoint $ADMLOPS_PATH/checkpoints/mmdet/yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth