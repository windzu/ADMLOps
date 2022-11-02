# Quick Test

## Train

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/mmdet_train.py $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py
```

### Resume

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/train.py $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
--resume-from $ADMLOPS/work_dirs/yolox_s_8x8_300e_coco-bdd10k/latest.pth
```

## Eval

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/local/configs/yolox/yolox_s_8x8_300e_coco-flir.py \
./work_dirs/yolox_s_8x8_300e_coco-flir/latest.pth \
--eval bbox
```

## ROS TEST

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/rosrun.py $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
./checkpoints/yolox/yolox_s_8x8_300e_coco.pth \
--topic /CAM_FRONT/image_rect_compressed \
--device cuda:0 \
--score_thr 0.3 \
--msg_type img \
--republish \
--compressed
```