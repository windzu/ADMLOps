# Quick Test

## Train

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/mmdet_train.py $ADMLOPS/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco-bdd10k.py
```

### Resume

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/mmdet_train.py $ADMLOPS/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco-bdd10k.py \
--resume-from $ADMLOPS/work_dirs/mask_rcnn_r50_fpn_1x_coco-bdd10k.py/latest.pth
```

## Eval

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco-bdd10k.py \
./work_dirs/mask_rcnn_r50_fpn_1x_coco-bdd10k/latest.pth \
--eval bbox
```

## ROS TEST

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/rosrun.py $ADMLOPS/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco-bdd10k.py \
./checkpoints/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.pth \
--topic /CAM_FRONT/image_rect_compressed \
--device cuda:0 \
--score_thr 0.3 \
--republish true \
--msg_type img \
--compressed_flag true
```