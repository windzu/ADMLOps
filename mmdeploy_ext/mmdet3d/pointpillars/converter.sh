python $MMDEPLOY_DIR/tools/deploy.py \
    $MMDEPLOY_DIR/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-kitti.py \
    $MMDETECTION3D_DIR/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py \
    $MMDETECTION3D_DIR/checkpoints/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
    $MMDETECTION3D_DIR/demo/data/kitti/kitti_000008.bin \
    --work-dir work_dir \
    --device cuda:0 \
    --show 