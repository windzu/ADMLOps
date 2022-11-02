import json
import numpy as np
import cv2
from urllib.request import urlopen
from rich.progress import track

import asyncio
from argparse import ArgumentParser

from mmdet.apis import async_inference_detector, inference_detector, init_detector, show_result_pyplot

COCO_TO_BDD100K = {
    "person": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}


class AutoAnnotation:
    def __init__(self, input, type, config, checkpoint, device, score_thr=0.3):
        self.input = input
        self.type = type
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.score_thr = score_thr

        # build the model from a config file and a checkpoint file
        self.model = init_detector(self.config, self.checkpoint, device=self.device)
        self.class_names = self.model.CLASSES

    def run(self):

        if str(self.type) == "scalabel":
            frames = self.parse_scalabel(self.input)
            for frame in track(frames):
                # test a single image
                # judge if the file is a path or url
                img_path = frame["url"]

                resp = urlopen(img_path)
                img = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(img, -1)

                result = inference_detector(self.model, img)
                (bboxes, labels, label_names) = self.format_result_to_standard_format(
                    result, self.class_names, self.score_thr
                )

                labels = self.format_result_to_scalabel_format(bboxes, labels, label_names)

                frame["labels"] = labels

            save_path = self.input.split(".json")[0] + "_auto_annotation.json"
            with open(save_path, "w") as f:
                json.dump(frames, f, indent=4)

            print("Save result to {}".format(save_path))

        elif str(self.type) == "voc":
            pass

        print("Done")

    def _filter_result(self, file, result):
        """通过mmdet的模型预测出的结果,过滤出符合要求的结果

        Args:
            result(dict): 检测结果
        """
        # 1. format the result to the standard format
        (bboxes, labels, label_names) = self.format_result_to_standard_format(result, self.class_names, self.score_thr)

        # 2. filter the result to the required format
        if str(self.type) == "scalabel":
            result = self.format_result_to_scalabel_format(bboxes, labels, label_names)
            if result:
                result["name"] = file
                result["url"] = file
                return result
            else:
                result = None
        else:
            print("Invalid type")
            return None

    def _parse_file_list(self):
        """根据输入的文件路径或者文件夹路径以及其对应的数据类型，解析出图片列表

        Returns:
            list(str): 带有图片路径的列表
        """
        if str(self.type) == "scalabel":
            return self.parse_scalabel(self.input)
        else:
            print("Invalid type")
            return None

    @staticmethod
    def parse_scalabel(path):
        """解析scalabel格式的图像列表文件

        scalabel格式的图像列表文件是一个json文件,其内容如下：
        [
            {
                "name": "http://localhost:8686/items/imgs_00/0000.jpg",
                "url": "http://localhost:8686/items/imgs_00/0000.jpg",
                "videoName": "",
                "timestamp": 0,
                "intrinsics":{
                    focal: [0, 0],
                    center: [0, 0],
                },
                "extrinsics":{
                    location: [0, 0, 0],
                    rotation: [0, 0, 0],
                },
                "attributes":{},
                "size":{
                    "width": 1920,
                    "height": 1080
                }
                "labels":[] # 等待自动标注进行填充
                "sensor": -1,
            },
            ...
        ]

        Args:
            input (str): scalabel格式的图像列表文件
        """
        # load json and get all url to list and return
        with open(path, "r") as f:
            frames = json.load(f)
            return frames

    @staticmethod
    def format_result_to_standard_format(result, class_names, score_thr=0.3):
        """将mmdet的检测结果转换成通用格式
        Args:
            result (Tensor or tuple): 检测结果
            class_names (list(str)): 类别名称列表
            score_thr (float): 分数阈值

        Returns:
            list(dict): 通用格式的检测结果

        """

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        # TODO : add support to segmentation masks

        # filter out the results
        assert bboxes is None or bboxes.ndim == 2, f" bboxes ndim should be 2, but its ndim is {bboxes.ndim}."
        assert labels.ndim == 1, f" labels ndim should be 1, but its ndim is {labels.ndim}."
        assert (
            bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        ), f" bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}."
        assert (
            bboxes is None or bboxes.shape[0] <= labels.shape[0]
        ), "labels.shape[0] should not be less than bboxes.shape[0]."

        if score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            # TODO : add support to filter segmentation masks

        # # debug
        # print("bboxes: ", bboxes)
        # print("labels: ", labels)

        # get label name
        label_names = [class_names[i] for i in labels]

        return (bboxes, labels, label_names)

    @staticmethod
    def format_result_to_scalabel_format(bboxes, labels, label_names):

        scalabel_labels = []
        # filter out the results which are not in the required format
        for i, (bbox, label, label_name) in enumerate(zip(bboxes, labels, label_names)):
            if label_name in COCO_TO_BDD100K:
                scalabel_labels.append(
                    {
                        "id": i,
                        "category": COCO_TO_BDD100K[label_name],
                        "attributes": {},
                        "manualShape": True,
                        "box2d": {
                            "x1": int(bbox[0]),
                            "y1": int(bbox[1]),
                            "x2": int(bbox[2]),
                            "y2": int(bbox[3]),
                        },
                        "poly2d": None,
                        "box3d": None,
                    }
                )

        if len(scalabel_labels) == 0:
            return []
        else:
            return scalabel_labels
