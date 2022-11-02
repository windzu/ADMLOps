conda activate mmdet && \
cd $MMLAB_EXTENSION_PATH/mmdetection && \
python demo/image_demo.py demo/demo.jpg \
    ./configs/yolox/yolox_tiny_8x8_300e_coco.py \
    ./checkpoints/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth \
    --device cuda:0 \
    --out-file result.jpg