# Quick Test

记录一下基本使用


## nuScenes
> 这里使用nuScenes_v1.0_mini进行演示,在此之前需要从数据集生成约定的pkl文件,具体参考文档中对应数据部分文档

### Train
```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/mmdet3d_train.py $ADMLOPS/configs/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py
```

### Resume

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/train.py $ADMLOPS/configs/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py \
--resume-from $ADMLOPS/work_dirs/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/latest.pth
```

### Eval

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/local/configs/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py \
./work_dirs/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/latest.pth \
--eval bbox
```

### ROS TEST

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/rosrun.py $ADMLOPS/configs/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py \
./checkpoints/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.pth \
--topic /LIDAR_TOP \
--device cuda:0 \
--score_thr 0.3 \
--msg_type pointcloud \
--republish
```

