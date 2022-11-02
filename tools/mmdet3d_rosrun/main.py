import sys
import os
from argparse import ArgumentParser
import numpy as np
import torch
from copy import deepcopy
import time


# add path
# get mmdetection3d path from env variable
admlops_path = os.environ["ADMLOPS_PATH"]

# mmlab
import mmcv
from mmcv.parallel import collate, scatter
from mmdet3d.apis import init_model
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type

# mmlab extension
from mmdet3d_extension.datasets import LoadPointsFromPointCloud2

# ros
from autoware_msgs.msg import DetectedObject, DetectedObjectArray
import rospy
from sensor_msgs.msg import PointCloud2


# local
from postprocess import result_process


class ROSExtension:
    """ROS的一个扩展,具体内容还未确定,首先先基于mmdet3d完成对点云处理的拓展"""

    def __init__(
        self,
        model,
        lidar_topic,
        detected_objects_topic,
        score_thr=0.1,
        republish=False,
    ):
        self.model = model
        self.lidar_topic = lidar_topic
        self.detected_objects_topic = detected_objects_topic
        self.score_thr = score_thr
        self.republish = republish

        self.device = next(self.model.parameters()).device
        self.cfg = self.model.cfg
        self.cfg = self.cfg.copy()

        ## build the data pipeline
        self.test_pipeline = deepcopy(self.cfg.data.test.pipeline)
        self.test_pipeline = Compose(self.test_pipeline)

        self.box_type_3d, self.box_mode_3d = get_box_type(self.cfg.data.test.box_type_3d)

    def start(self):
        rospy.init_node("lidar_detection", anonymous=True)
        # detected objects publisher
        self.detected_objects_publisher = rospy.Publisher(
            self.detected_objects_topic, DetectedObjectArray, queue_size=1
        )

        # if republish lidar for sync pointcloud2 and detected objects
        self.republish_topic = self.lidar_topic + "/republish"
        self.lidar_republish_publisher = rospy.Publisher(self.republish_topic, PointCloud2, queue_size=1)

        # subscribe lidar topic
        self.lidar_subscriber = rospy.Subscriber(
            self.lidar_topic, PointCloud2, self.__callback, queue_size=1, buff_size=2**24
        )
        rospy.spin()

    def __callback(self, msg):
        # 1. prepare data
        data = self.__create_data(msg)
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)

        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device.index])[0]
        else:
            # this is a workaround to avoid the bug of MMDataParallel
            data["img_metas"] = data["img_metas"][0].data
            data["points"] = data["points"][0].data

        # 2. forward the model
        start_time = time.time()
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)
        end_time = time.time()
        print("inference time: {}".format(end_time - start_time))

        # 3. postprocess
        detected_object_array = result_process(
            result=results[0],
            score_thr=self.score_thr,
            CLASSES=self.model.CLASSES,
            frame_id=msg.header.frame_id,
        )
        self.detected_objects_publisher.publish(detected_object_array)

        # 4. republish
        if self.republish:
            self.lidar_republish_publisher.publish(msg)

    def __create_data(self, msg):
        data = dict(
            pointcloud2=msg,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            # for ScanNet demo we need axis_align_matrix
            ann_info=dict(axis_align_matrix=np.eye(4)),
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[],
        )
        return data


def quick_combine_config(test_mode, model_name, config_file_name):
    """快速组合config
    在rosrun的debug模式下,经常需要测试多种模型的效果,以及测试训练好的模型在特定场景下的效果
    在构建模型的时候其配置参数的格式、checkpoint的存放路径等都是相对固定的,所以本函数的作用就是
    通过简单的输入参数来快速组合config
    Args:
        test_mode (bool): 是否为测试模式,测试模式下加载的是训练好的模型,否则加载的是预训练模型
        model_name (str): 模型网络名称,如"pointpillars", "centerpoint"
        config_file_name (str): 模型的配置文件名称,这个部分比较复杂,需要自己去维护一个常用的配置文件名称的字典
    Returns:
        tuple: (config_path, checkpoint_path):
            config_path (str): 模型的配置文件路径
            checkpoint_path (str): 模型的checkpoint路径
    """
    config_path = os.path.join(
        admlops_path,
        "mmdetection3d_extension",
        "configs",
        model_name,
        config_file_name,
    )
    if test_mode:
        checkpoint_path = os.path.join(
            admlops_path,
            "mmdetection3d_extension",
            "work_dirs",
            config_file_name.split(".")[0],
            "latest.pth",
        )
    else:
        current_checkpoints_dir = os.path.join(
            admlops_path,
            "mmdetection3d",
            "checkpoints",
            model_name,
        )
        # find which checkpoint name is start with config_file_name
        checkpoints_list = os.listdir(current_checkpoints_dir)
        checkpoints = [
            checkpoint for checkpoint in checkpoints_list if checkpoint.startswith(config_file_name.split(".")[0])
        ]
        if len(checkpoints) == 0:
            raise FileNotFoundError("checkpoint not found")
        checkpoint_path = os.path.join(current_checkpoints_dir, checkpoints[0])
    return config_path, checkpoint_path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint file path")
    parser.add_argument("--topic", help="ros lidar topic")
    parser.add_argument("--score_thr", type=float, default=0.1, help="bbox score threshold")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--republish", type=bool, default=True, help="if republish lidar for sync pointcloud2 ")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline[0].type = "LoadPointsFromPointCloud2"
    config.data.test.pipeline[0]["load_dim"] = 5
    config.data.test.pipeline[0]["use_dim"] = 5

    # init model
    model = init_model(config, args.checkpoint, device=args.device)
    print("---model init done---")

    ros_extension = ROSExtension(
        model=model,
        detected_objects_topic=args.topic + "/detected_objects",
        lidar_topic=args.topic,
        score_thr=args.score_thr,
        republish=args.republish,
    )
    print("---waiting for topic %s msgs---:" % args.topic)
    ros_extension.start()


if __name__ == "__main__":
    main()
