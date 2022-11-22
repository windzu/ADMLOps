## 模型转换

**动态 tensorrt**

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

## 模型评估

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
