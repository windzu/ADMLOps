# Quick Test

## COCO_BDD10K
### Train

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/mmdet_train.py $ADMLOPS/configs/yolopv2/yolopv2_s_8x8_300e_coco-bdd10k.py
```

### Resume

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/train.py $ADMLOPS/configs/yolopv2/yolopv2_s_8x8_300e_coco-bdd10k.py \
--resume-from $ADMLOPS/work_dirs/yolopv2_s_8x8_300e_coco-bdd10k/latest.pth
```

### Eval

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/local/configs/yolopv2/yolopv2_s_8x8_300e_coco-bdd10k.py \
./work_dirs/yolopv2_s_8x8_300e_coco-bdd10k/latest.pth \
--eval bbox
```

### ROS TEST

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/rosrun.py $ADMLOPS/configs/yolopv2/yolopv2_s_8x8_300e_coco-bdd10k.py \
./checkpoints/yolopv2/yolopv2_s_8x8_300e_coco-bdd10k.pth \
--topic /CAM_FRONT/image_rect_compressed \
--device cuda:0 \
--score_thr 0.3 \
--msg_type img \
--republish \
--compressed
```