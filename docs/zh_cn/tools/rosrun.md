# ROS RUN

rosrun是一个测试模型的简单接口，通过ros输入数据，使用python调用模型推理，将结果通过ros输出。这样可以方便的测试某一个模型在某一个场景下的效果。

## 使用说明

- --config：模型的配置文件路径
- --checkpoint：模型的权重文件路径
- --score_thr：过滤结果的置信度阈值，范围\[0, 1)
- --device：推理使用的设备，`cpu`或`cuda:x`，x为GPU的编号
- --task_type：模型的任务类型，支持的任务类型有`det2d`,`seg2d`,`det3d`,`seg3d`
- --sub_topic：输入数据的ros topic
- --sub_msg_type：输入数据的类型，支持的类型有`img`,`pc`，`img`表示输入的是图像，`pc`表示输入的是点云
- --republish：启用该选项后，会将输入的数据通过ros重新发布一遍，用于一些延迟较高的场景，以保证推理的"实时性"
- --compressed：用于显式的告知输入的数据是`img`的`compressed`类型，所以仅对`img`类型的输入有效

## 示例

### 2D检测

测试yolox_s_8x8_300e_coco模型在ros中的效果

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/rosrun.py --config $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco.py \
--checkpoint ./checkpoints/yolox/yolox_s_8x8_300e_coco.pth \
--score_thr 0.3 \
--device cuda:0 \
--task_type det2d \
--sub_topic /CAM_FRONT/image_rect_compressed \
--sub_msg_type img \
--republish \
--compressed
```

### 3D检测

测试hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class模型在ros中的效果

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
