export model_type=pointpillars && \
export model_name=hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class && \
python $ADMLOPS_PATH/mmdeploy/tools/deploy.py \
    $ADMLOPS_PATH/mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-kitti-32x4.py \
    $ADMLOPS_PATH/mmdetection3d_extension/configs/$model_type/$model_name.py \
    $ADMLOPS_PATH/mmdetection3d_extension/work_dirs/$model_name/latest.pth \
    $ADMLOPS_PATH/mmdetection3d_extension/demo/data/kitti/kitti_000008.bin\
    --work-dir $ADMLOPS_PATH/mmdeploy_extension/mmdet3d/$model_type/$model_name/work_dir \
    --device cuda:0