export model_type=pointpillars && \
export model_name=hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class && \
python $ADMLOPS_PATH/mmdeploy/tools/deploy.py \
    $ADMLOPS_PATH/mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus-20x5.py \
    $ADMLOPS_PATH/mmdetection3d_extension/configs/$model_type/$model_name.py \
    --speed-test \
    --model $ADMLOPS_PATH/mmdeploy_extension/mmdet3d/$model_type/$model_name/work_dir/end2end.engine \
    --device cuda:0
