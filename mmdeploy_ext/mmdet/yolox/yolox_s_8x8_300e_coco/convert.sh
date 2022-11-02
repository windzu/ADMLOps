export model_type=yolox && \
export model_name=yolox_s_8x8_300e_coco && \
python $ADMLOPS_PATH/mmdeploy/tools/deploy.py \
    $ADMLOPS_PATH/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    $ADMLOPS_PATH/mmdetection_extension/configs/$model_type/$model_name.py \
    $ADMLOPS_PATH/mmdetection_extension/work_dirs/$model_name/latest.pth \
    $ADMLOPS_PATH/mmdetection_extension/demo/demo.jpg \
    --test-img $ADMLOPS_PATH/mmdetection_extension/demo/demo.jpg \
    --work-dir $ADMLOPS_PATH/mmdeploy_extension/mmdet/$model_type/$model_name/work_dir \
    --device cuda:0 \
    --dump-info