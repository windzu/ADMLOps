import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

# opem-mmlab
import mmcv
import cv2

# ros
import rospy
from autoware_msgs.msg import DetectedObject, DetectedObjectArray


def result_process(result, score_thr, CLASSES, frame_id="map"):
    """将网络的检测结果后处理为autoware_msgs/DetectedObjectArray格式

    Args:
        result (list): 网络的检测结果 [[x1,y1,x2,y2,score],...] x1,y1为左上角坐标,x2,y2为右下角坐标
        score_thr (float):
        CLASSES (list): 本模型的类别列表

    Returns:
        detected_objects(DetectedObjectArray): 转换后格式的检测结果
    """

    #     bboxes, labels, _ = detector([img])[0]
    #
    #     indices = [i for i in range(len(bboxes))]
    #     for index, bbox, label_id in zip(indices, bboxes, labels):
    #         [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
    #         if score < 0.3:
    #             continue
    #         cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

    if isinstance(result, tuple):
        bboxes, labels, _ = result
    else:
        return None

    # indices = [i for i in range(len(bboxes))]
    # for index, bbox, label_id in zip(indices, bboxes, labels):
    #     [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
    #     if score < score_thr:
    #         continue
    #     cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

    # bboxes = np.vstack(bbox_result)
    # labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    # labels = np.concatenate(labels)
    # # draw segmentation masks
    # # NOTE : 暂时不支持分割结果

    # filter results
    (bboxes, labels, class_names) = filter_det_result(
        bboxes=bboxes,
        labels=labels,
        CLASSES=CLASSES,
        score_thr=score_thr,
    )

    # convert result to DetectedObjectArray
    detected_objects = DetectedObjectArray()
    detected_objects.header.stamp = rospy.Time.now()
    detected_objects.header.frame_id = frame_id
    for i in range(len(bboxes)):
        detected_object = DetectedObject()
        detected_object.header.stamp = rospy.Time.now()
        detected_object.header.frame_id = frame_id
        detected_object.label = class_names[i]
        detected_object.score = bboxes[i][4]
        detected_object.x = int(bboxes[i][0])
        detected_object.y = int(bboxes[i][1])
        detected_object.width = int(bboxes[i][2]) - int(bboxes[i][0])
        detected_object.height = int(bboxes[i][3]) - int(bboxes[i][1])

        detected_objects.objects.append(detected_object)

    return detected_objects


def filter_det_result(bboxes=None, labels=None, CLASSES=None, score_thr=0):
    """parse bboxes and class labels (with scores) on an image.

    Args:
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.

    Returns:
        tuple(ndarray,ndarray,list[str]): The image with bboxes drawn on it.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or (n, 5).
            labels (ndarray): Labels of bboxes.
            class_names (list[str]): Names of each classes.
    """
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
        class_names = [CLASSES[label] for label in labels.tolist()]

    return (bboxes, labels, class_names)
