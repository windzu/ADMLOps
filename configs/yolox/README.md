# Quick Test

## COCO_BDD10K
### Train

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

### Eval

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/local/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
./work_dirs/yolox_s_8x8_300e_coco-bdd10k/latest.pth \
--eval bbox
```

### ROS TEST

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/rosrun.py --config $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
--checkpoint ./checkpoints/yolox/yolox_s_8x8_300e_coco.pth \
--score_thr 0.3 \
--device cuda:0 \
--task_type det2d \
--sub_topic /CAM_FRONT/image_rect_compressed \
--sub_msg_type img \
--republish \
--compressed
```