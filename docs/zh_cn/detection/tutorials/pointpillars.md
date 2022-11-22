# YOLOX

本教程将演示YOLOX的使用,共包然以下几部分内容:

- 准备标准数据集
- 在标准数据集上训练已有模型
- 在标准数据集上训练自定义模型
- 模型推理测试
- 模型部署

## 准备标准数据集

YOLOX是一个目标检测模型，它的输入是图片，输出是检测框和类别, 可以使用mmdetection原生支持的COCO数据集进行训练，而且可以使用COCO数据集的评估标准进行评估。

本教程以COCO数据集为例，使用BDD100K转COCO，准备过程请参考[BDD100K TO COCO](../../dataset/bdd100k.md),将BDD100K中的BDD10K数据集的instances_train转换为COCO格式

数据的组织结构如下：

```bash
COCO_BDD10K
├── train # 原BDD100K中BDD10K的train数据集
├── test # 原BDD100K中BDD10K的test数据集
├── val # 原BDD100K中BDD10K的test数据集
├── annotations # 转换后数据集标注文件
│   ├── instances_train.json
│   ├── instances_val.json
│   ├── instances_test.json
```

## 在标准数据集上训练已有模型

完成数据集准备后，可以使用已有模型进行训练，本教程数据集使用上步骤转换后的[COCO_BDD10K](../../../../data/COCO_BDD10K/),模型采用YOLOX-S为例进行训练，模型配置文件见[config](../../../../configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py),对于网络结构的详细接受参考其注释即可

**train**

训练命令如下，训练完成后会在`work_dirs`下生成`yolox_s_8x8_300e_coco-bdd10k`文件夹，该文件夹下包含训练过程中的模型文件，训练日志等文件

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/mmdet_train.py $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py
```

**resume train**

如果训练中断，可以使用`resume_from`参数加载中断时的模型文件继续训练，训练命令如下

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/mmdet_train.py $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
--resume-from $ADMLOPS/work_dirs/yolox_s_8x8_300e_coco-bdd10k/latest.pth
```

## 在标准数据集上训练自定义模型

如果想要训练自定义的模型，可以参考[yolopv2](./yolopv2.md)的使用文档，yolopv2是在yolox的基础上进行了修改而实现的

## 模型推理测试

**ros test**

使用ros对训练好的模型进行推理测试，需要指定模型文件和测试相关的参数，具体指令如下，更多关于rosrun的使用请参考[rosrun](../../tools/rosrun.md)

```bash
conda activate ADMLOps && \
cd $ADMLOPS && \
python3 tools/rosrun.py --config $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
--checkpoint ./work_dirs/yolox_s_8x8_300e_coco-bdd10k/latest.pth \
--score_thr 0.3 \
--device cuda:0 \
--task_type det2d \
--sub_topic /CAM_FRONT/image_rect_compressed \
--sub_msg_type img \
--republish \
--compressed
```

## 模型部署

因为部署模型部分跟进的比较慢，所以关于部署的文档将单独进行介绍，请移步参考对应的[部署文档](../../deploy/tutorials/pointpillars.md)
