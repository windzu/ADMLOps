import numpy as np
from scipy.spatial.transform import Rotation as R

# ros
import rospy
from autoware_msgs.msg import DetectedObject, DetectedObjectArray


def result_process(result, score_thr, CLASSES, frame_id="map"):
    """将网络的检测结果后处理为autoware_msgs/DetectedObjectArray格式

    Args:
        result (list): 网络的检测结果
        score_thr (float):
        CLASSES (list): 本模型的类别列表

    Returns:
        detected_objects(DetectedObjectArray): 转换后格式的检测结果
    """
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


