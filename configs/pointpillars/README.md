# Quick Test

记录一下基本使用

## nuScenes

> 这里使用nuScenes_v1.0_mini进行演示

### Data Preparation

> 在此之前需要从数据集生成约定的pkl文件，如果已生成则可继续下一步骤即可(这个步骤比较消耗时间)

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/create_data.py nuscenes --root-path ./data/nuScenes_v1.0_mini \
--out-dir ./data/nuScenes_v1.0_mini \
--extra-tag nuscenes
```

### Train

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/mmdet3d_train.py $ADMLOPS/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py
```

### Resume

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/train.py $ADMLOPS/local/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py \
--resume-from $ADMLOPS/work_dirs/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/latest.pth
```

### Eval

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/local/configs/yolox/yolox_s_8x8_300e_coco-flir.py \
./work_dirs/yolox_s_8x8_300e_coco-flir/latest.pth \
--eval bbox
```

### ROS TEST

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/rosrun.py --config $ADMLOPS/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py \
--checkpoint ./checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.pth \
--score_thr 0.3 \
--device cuda:0 \
--task_type det3d \
--sub_topic /LIDAR_TOP \
--sub_msg_type pc \
--republish
```

## KITTI

### Data Preparation

> 在此之前需要从数据集生成约定的pkl文件，如果已生成则可继续下一步骤即可(这个步骤比较消耗时间)

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python tools/create_data.py kitti --root-path ./data/KITTI --out-dir ./data/KITTI --extra-tag kitti
```

### Train

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/mmdet3d_train.py $ADMLOPS/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py
```

### Resume

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/train.py $ADMLOPS/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py \
--resume-from $ADMLOPS/work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/latest.pth
```

### Eval

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/local/configs/yolox/yolox_s_8x8_300e_coco-flir.py \
./work_dirs/yolox_s_8x8_300e_coco-flir/latest.pth \
--eval bbox
```

### ROS TEST

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/rosrun.py --config $ADMLOPS/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py \
--checkpoint ./checkpoints/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.pth \
--score_thr 0.3 \
--device cuda:0 \
--task_type det3d \
--sub_topic /LIDAR_TOP \
--sub_msg_type pc \
--republish
```
