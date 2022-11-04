# Copyright (c) OpenMMLab. All rights reserved.
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

# ros
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from autoware_msgs.msg import DetectedObject, DetectedObjectArray

# mmcv
import mmcv

# mmdet
from mmdet.models import build_detector
from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor

# mmdet3d
from mmdet3d.apis import init_model

# mmseg
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot

# local
from ros_utils import ROSExtension
from mmdet_ext.models import *
from mmdet_ext.datasets import *
from mmdet3d_ext.datasets import *


class ROSInterface:
    def __init__(
        self,
        config,
        checkpoint,
        score_thr,
        device,
        task_type,
        sub_topic,
        sub_msg_type,
        republish=False,
        compressed=False,
    ):
        # about model
        self.config = config
        self.checkpoint = checkpoint
        self.score_thr = score_thr
        self.device = device
        self.task_type = task_type
        # about ros
        self.sub_topic = sub_topic
        self.sub_msg_type = sub_msg_type
        self.republish = republish
        self.compressed = compressed

        # 根据task_type,初始化不同的模型，推理函数，后处理函数
        # -self.model
        # -self.inference
        # -self.postprocess
        if self.task_type == "det2d":
            from mmdet.apis import init_detector, inference_detector
            from ros_utils.postprocess import det2d_postprocess

            self.model = init_detector(self.config, self.checkpoint, device=self.device)
            self.inference = inference_detector
            self.postprocess = det2d_postprocess
        elif self.task_type == "seg2d":
            from mmseg.apis import init_segmentor, inference_segmentor
            from ros_utils.postprocess import seg2d_postprocess

            self.model = init_segmentor(self.config, self.checkpoint, device=self.device)
            self.inference = inference_segmentor
            self.postprocess = seg2d_postprocess
        elif self.task_type == "det3d":
            from mmdet3d.apis import init_model
            from mmdet3d_ext.apis import inference_detector  # 重写了inference_detector
            from ros_utils.postprocess import det3d_postprocess

            self.model = init_model(self.config, self.checkpoint, device=self.device)
            # 替换LoadPointsFromFile
            self.model.cfg.data.test.pipeline[0].type = "LoadPointsFromPointCloud2"
            self.inference = inference_detector
            self.postprocess = det3d_postprocess

    def start(self):
        rospy.init_node("detection", anonymous=True)

        self.publisher = rospy.Publisher(
            self.sub_topic + "/detected_objects", DetectedObjectArray, queue_size=1
        )

        # 根据消息类型的不同,确定Subscriber的消息类型
        if self.sub_msg_type == "img":
            if self.compressed:
                rospy.Subscriber(
                    self.sub_topic, CompressedImage, self._callback, queue_size=1, buff_size=2**24
                )
                self.repub_publisher = rospy.Publisher(
                    self.sub_topic + "/republish", CompressedImage, queue_size=1
                )
            else:
                rospy.Subscriber(
                    self.sub_topic, Image, self._callback, queue_size=1, buff_size=2**24
                )
                self.repub_publisher = rospy.Publisher(
                    self.sub_topic + "/republish", Image, queue_size=1
                )
        elif self.sub_msg_type == "pc":
            rospy.Subscriber(
                self.sub_topic, PointCloud2, self._callback, queue_size=1, buff_size=2**24
            )
            self.repub_publisher = rospy.Publisher(
                self.sub_topic + "/republish", PointCloud2, queue_size=1
            )
        else:
            raise ValueError("msg_type must be img or pointcloud")

        rospy.spin()

    def _callback(self, msg):
        # 1. preprocess msg
        if self.sub_msg_type == "img":
            if self.compressed:
                buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
                data = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            else:
                data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        elif self.sub_msg_type == "pc":
            data = msg

        # 2. inference
        start_time = time.time()
        result = self.inference(self.model, data)
        end_time = time.time()
        print("inference time: {}".format(end_time - start_time))

        # 3. postprocess
        result = self.postprocess(result, self.score_thr, self.model.CLASSES, msg.header.frame_id)

        # 4. publish result
        self.publisher.publish(result)


def parse_args():
    parser = ArgumentParser()
    # about model
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--checkpoint", help="model checkpoint file")
    parser.add_argument("--score_thr", type=float, default=0.0, help="bbox score threshold")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--task_type", type=str, help="task type, det2d or seg2d or det3d")
    # about ros
    parser.add_argument("--sub_topic", help="msg's rostopic")
    parser.add_argument("--sub_msg_type", type=str, help="msg type, img or pc")
    parser.add_argument("--republish", action="store_true", help="if republish the original msg")
    parser.add_argument("--compressed", action="store_true", help="if compressed image")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ros_interface = ROSInterface(
        config=args.config,
        checkpoint=args.checkpoint,
        score_thr=args.score_thr,
        device=args.device,
        task_type=args.task_type,
        sub_topic=args.sub_topic,
        sub_msg_type=args.sub_msg_type,
        republish=args.republish,
        compressed=args.compressed,
    )
    print("---waiting for topic %s msgs---:" % args.sub_topic)
    ros_interface.start()


if __name__ == "__main__":
    main()
