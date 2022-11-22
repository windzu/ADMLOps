# Quick Test

## COCO_BDD10K

### Train

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/mmdet_train.py $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py
```

### Resume

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/train.py $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
--resume-from $ADMLOPS/work_dirs/yolox_s_8x8_300e_coco-bdd10k/latest.pth
```

### Eval

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/local/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
./work_dirs/yolox_s_8x8_300e_coco-bdd10k/latest.pth \
--eval bbox
```

### ROS TEST

```bash
conda activate ADMLOps && \
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

### Deploy

**convert to tensorrt**

```bash
export model_type=yolox && \
export model_name=yolox_s_8x8_300e_coco && \
cd $ADMLOPS && \
python ./tools/deploy.py \
    ./configs/deploy/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ./configs/$model_type/$model_name.py \
    ./mmdetection_extension/work_dirs/$model_name/latest.pth \
    ./mmdetection_extension/demo/demo.jpg \
    --test-img ./demo/demo.jpg \
    --work-dir ./work_dirs/deploy/$model_name \
    --device cuda:0 \
    --dump-info
```

**eval tensorrt**

```bash
export model_type=yolox && \
export model_name=yolox_s_8x8_300e_coco && \
cd $ADMLOPS && \
python ./tools/deploy_test.py \
    ./configs/deploy/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ./configs/$model_type/$model_name.py \
    --speed-test \
    --model ./work_dirs/deploy/$model_name/end2end.engine \
    --device cuda:0
```

**ros run test**

```bash
on the way
```
