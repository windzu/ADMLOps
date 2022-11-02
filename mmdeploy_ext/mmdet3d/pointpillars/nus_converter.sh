# # use secfpn
# python $MMDEPLOY_DIR/tools/deploy.py \
#     $MMDEPLOY_DIR/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus.py \
#     $MMDETECTION3D_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py \
#     $MMDETECTION3D_DIR/checkpoints/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth \
#     $MMDETECTION3D_DIR/demo/data/kitti/kitti_000008_5d.bin \
#     --work-dir work_dir/nus \
#     --device cuda:0 \
#     --show 

# use fpn
python $MMDEPLOY_DIR/tools/deploy.py \
    $MMDEPLOY_DIR/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus.py \
    $MMDETECTION3D_DIR/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py \
    $MMDETECTION3D_DIR/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth \
    $MMDETECTION3D_DIR/demo/data/kitti/kitti_000008_5d.bin \
    --work-dir work_dir/nus \
    --device cuda:0 \
    --show 