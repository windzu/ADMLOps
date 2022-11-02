import asyncio
from argparse import ArgumentParser


import sys
import os
from argparse import ArgumentParser
import numpy as np
import torch
from copy import deepcopy
import time
import cv2


# mmcv
import mmcv
from mmcv.parallel import collate, scatter

# mmdet
from mmdet.datasets import replace_ImageToTensor

# mmdet3d
from mmdet3d.core.bbox import get_box_type

# from mmdet3d.datasets.pipelines import Compose  # mmdet3d compose 兼容 mmdet
# from mmdet.datasets.pipelines import Compose


# ros
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from autoware_msgs.msg import DetectedObject, DetectedObjectArray

# local
from .postprocess import result_process


class ROSExtension:
    """ROS的一个扩展
    Args:
        model (nn.Module): 模型
        sub_topic (str): 订阅的消息的topic,模型检测该topic内容
        pub_topic (str): 发布检测结果的topic
        score_thr (float, optional): 检测结果的阈值. Defaults to 0.1.
        msg_type (str, optional): 消息的类型,目前支持 img 和 pointcloud. Defaults to "img".
        republish (bool, optional): 是否需要重复发布原始的消息. Defaults to False.
        compressed (bool, optional): 是否使用压缩的图像. Defaults to False.
    """

    def __init__(
        self,
        model,
        sub_topic,
        pub_topic,
        score_thr=0.1,
        msg_type="img",
        republish=False,
        compressed=False,
    ):
        self.model = model
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.score_thr = score_thr
        self.republish = republish
        self.msg_type = msg_type
        self.compressed = compressed

        # 从model中获取一些信息
        self.device = next(self.model.parameters()).device

        ## build the data pipeline
        self.cfg = deepcopy(self.model.cfg)

        if self.msg_type == "img":
            self.cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
            self.cfg.data.test.pipeline = replace_ImageToTensor(
                self.cfg.data.test.pipeline
            )

            # use mmdet compose
            from mmdet.datasets.pipelines import Compose

            self.pipeline = Compose(self.cfg.data.test.pipeline)
        elif self.msg_type == "pointcloud":
            self.cfg.data.test.pipeline[0].type = "LoadPointsFromPointCloud2"
            self.cfg.data.test.pipeline[0].load_dim = 5
            self.cfg.data.test.pipeline[0].use_dim = 5
            self.box_type_3d, self.box_mode_3d = get_box_type(
                self.cfg.data.test.box_type_3d
            )

            # use mmdet compose
            from mmdet3d.datasets.pipelines import Compose

            self.pipeline = Compose(self.cfg.data.test.pipeline)

        # 变量声明
        self.publisher = None
        self.repub_publisher = None

    def start(self):
        rospy.init_node("detection", anonymous=True)

        # init publisher : publisher detected objects which is detected
        self.publisher = rospy.Publisher(
            self.pub_topic, DetectedObjectArray, queue_size=1
        )

        # init publisher : republish raw msg for sync msg and detected result
        self.repub_topic = self.sub_topic + "/republish"

        # init subscriber
        if self.msg_type == "img":
            if self.compressed:
                rospy.Subscriber(
                    self.sub_topic,
                    CompressedImage,
                    self._callback,
                    queue_size=1,
                    buff_size=2**24,
                )
                self.repub_publisher = rospy.Publisher(
                    self.repub_topic, CompressedImage, queue_size=1
                )
            else:
                rospy.Subscriber(
                    self.sub_topic,
                    Image,
                    self._callback,
                    queue_size=1,
                    buff_size=2**24,
                )
                self.repub_publisher = rospy.Publisher(
                    self.repub_topic, Image, queue_size=1
                )
        elif self.msg_type == "pointcloud":
            rospy.Subscriber(
                self.sub_topic,
                PointCloud2,
                self._callback,
                queue_size=1,
                buff_size=2**24,
            )
            self.repub_publisher = rospy.Publisher(
                self.repub_topic, PointCloud2, queue_size=1
            )
        else:
            raise ValueError("msg_type must be img or pointcloud")

        rospy.spin()

    def _callback(self, msg):

        if self.msg_type == "img":
            if self.compressed:
                buf = np.ndarray(
                    shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data
                )
                img = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, -1
                )

            data = dict(img=img)
            data = self.pipeline(data)
            data = collate([data], samples_per_gpu=1)
            # just get the actual data from DataContainer
            data["img_metas"] = [
                img_metas.data[0] for img_metas in data["img_metas"]
            ]
            data["img"] = [img.data[0] for img in data["img"]]
            data = scatter(data, [self.device])[0]
        elif self.msg_type == "pointcloud":
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
            data = self.pipeline(data)
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
            msg_type=self.msg_type,
        )
        self.publisher.publish(detected_object_array)

        # 4. republish
        if self.republish:
            repub_msg = deepcopy(msg)
            repub_msg.header.stamp = rospy.Time.now()
            self.repub_publisher.publish(repub_msg)
