import numpy as np
from scipy.spatial.transform import Rotation as R

# opem-mmlab
import mmcv
import cv2

# ros
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from autoware_msgs.msg import DetectedObject, DetectedObjectArray


def det2d_postprocess(result, score_thr, CLASSES, frame_id="map"):
    """对 2d det 结果进行后处理, 返回autoware_msgs.msg.DetectedObjectArray

    Args:
        result (tuple): 2d bbox检测结果
        score_thr (float): 分数阈值
        CLASSES (list[str]): _description_
        frame_id (str, optional): 消息传感器的frame_id. Defaults to "map".
    """

    def filter_det_result(bboxes=None, labels=None, CLASSES=None, score_thr=0):
        """parse bboxes and class labels (with scores) on an image.

        Args:
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or (n, 5).
            labels (ndarray): Labels of bboxes.
            CLASSES (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown. Default: 0.

        Returns:
            tuple(ndarray,ndarray,list[str]): The image with bboxes drawn on it.
                bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or (n, 5).
                labels (ndarray): Labels of bboxes.
                class_names (list[str]): Names of each classes.
        """
        assert (
            bboxes is None or bboxes.ndim == 2
        ), f" bboxes ndim should be 2, but its ndim is {bboxes.ndim}."
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

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
            # NOTE : 暂时不支持分割结果
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)
    # draw segmentation masks
    # NOTE : 暂时不支持分割结果

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


def seg2d_postprocess(result, score_thr, CLASSES, frame_id="map"):
    """对 2d seg 结果进行后处理, 返回autoware_msgs.msg.DetectedObjectArray

    Args:
        result (Tensor): 分割结果,shape为(H, W),每个像素点的数值对应一个类别的id
        score_thr (_type_): _description_
        CLASSES (_type_): _description_
        frame_id (str, optional): _description_. Defaults to "map".
    """

    seg = result[0]
    seg = np.array(seg)
    seg = seg.astype(np.uint8)
    seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    # show result
    cv2.imshow("seg", seg)
    cv2.waitKey(1)

    # TODO : convert result to DetectedObjectArray
    detected_objects = DetectedObjectArray()

    return detected_objects


def det3d_postprocess(result, score_thr, CLASSES, frame_id="map"):
    """对 3d det 结果进行后处理, 返回autoware_msgs.msg.DetectedObjectArray

    Args:
        result (_type_): _description_
        score_thr (_type_): _description_
        CLASSES (_type_): _description_
        frame_id (str, optional): _description_. Defaults to "map".

    Returns:
        _type_: _description_
    """

    # 检测的结果包含两个部分：results, data
    results, data = result
    # results是一个list, 每个元素是一个dict, 对应一个输入的检测结果
    result = results[0]

    if "pts_bbox" in result.keys():
        pred_bboxes = result["pts_bbox"]["boxes_3d"].tensor.numpy()
        pred_scores = result["pts_bbox"]["scores_3d"].numpy()
        pred_labels = result["pts_bbox"]["labels_3d"].numpy()
    else:
        pred_bboxes = result["boxes_3d"].tensor.numpy()
        pred_scores = result["scores_3d"].numpy()
        pred_labels = result["labels_3d"].numpy()

    # filter the result
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    # create autoware_msgs.msg.DetectedObjectArray
    detected_object_array = DetectedObjectArray()
    detected_object_array.header.frame_id = frame_id
    detected_object_array.header.stamp = rospy.Time.now()

    for i in range(len(pred_bboxes)):
        pred_bbox = pred_bboxes[i]
        # 前个数 分别对应xyz lwh yaw
        detected_object = DetectedObject()
        detected_object.header.frame_id = frame_id
        detected_object.header.stamp = rospy.Time.now()

        # 记录socre、label
        detected_object.score = pred_scores[i]
        detected_object.label = CLASSES[pred_labels[i]]

        # valid etc
        detected_object.valid = True
        detected_object.pose_reliable = True

        # xyz
        detected_object.pose.position.x = pred_bbox[0]
        detected_object.pose.position.y = pred_bbox[1]
        detected_object.pose.position.z = pred_bbox[2]

        # lwh
        detected_object.dimensions.x = pred_bbox[3]
        detected_object.dimensions.y = pred_bbox[4]
        detected_object.dimensions.z = pred_bbox[5]

        # yaw
        yaw = pred_bbox[6]

        # 高度修正
        detected_object.pose.position.z += detected_object.dimensions.z / 2

        # orientation
        # get quaternion from yaw axis euler angles
        r = R.from_euler("z", yaw, degrees=False)
        q = r.as_quat()
        detected_object.pose.orientation.x = q[0]
        detected_object.pose.orientation.y = q[1]
        detected_object.pose.orientation.z = q[2]
        detected_object.pose.orientation.w = q[3]

        detected_object_array.objects.append(detected_object)
    return detected_object_array
