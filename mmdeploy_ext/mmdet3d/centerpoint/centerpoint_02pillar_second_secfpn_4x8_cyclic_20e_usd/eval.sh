export model_type=yolox && \
export model_name=yolox_s_8x8_300e_coco && \
python $ADMLOPS_PATH/mmdeploy/tools/test.py \
    $ADMLOPS_PATH/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    $ADMLOPS_PATH/mmdetection_extension/configs/$model_type/$model_name.py \
    --speed-test \
    --model $ADMLOPS_PATH/mmdeploy_extension/mmdet/$model_type/$model_name/work_dir/end2end.engine \
    --device cuda:0 