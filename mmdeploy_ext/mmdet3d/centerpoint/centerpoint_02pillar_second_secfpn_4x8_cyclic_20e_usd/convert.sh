export model_type=centerpoint && \
export model_name=centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_usd && \
python $ADMLOPS_PATH/mmdeploy/tools/deploy.py \
    $ADMLOPS_PATH/mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus-20x5.py \
    $ADMLOPS_PATH/mmdetection3d_extension/configs/$model_type/$model_name.py \
    $ADMLOPS_PATH/mmdetection3d_extension/work_dirs/$model_name/latest.pth \
    $ADMLOPS_PATH/mmdetection3d_extension/demo/data/kitti/kitti_000008_5d.bin\
    --work-dir $ADMLOPS_PATH/mmdeploy_extension/mmdet3d/$model_type/$model_name/work_dir \
    --device cuda:0