# system
import os
import time
import pprint
from argparse import ArgumentParser
import torch
from torch.nn import functional as F
import onnxruntime as ort
import numpy as np
from scipy.spatial.transform import Rotation as R

# open mmlab
import mmcv
from mmcv.ops import Voxelization
from mmdet.core import build_prior_generator, build_bbox_coder
from mmdet3d.core.bbox import get_box_type, limit_period

# local class
from voxel_generator import VoxelGenerator
from anchor_3d_generator import AlignedAnchor3DRangeGenerator
from delta_xyzwhlr_bbox_coder import DeltaXYZWLHRBBoxCoder
from box3d_nms import box3d_multiclass_nms


# ros
import rospy
import ros_numpy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from autoware_msgs.msg import DetectedObject, DetectedObjectArray

# pypcd
from pypcd import pypcd
from pypcd import numpy_pc2


class PointPillarsDetector:
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path
        self.tensor_voxel_layer = Voxelization(**self.config.model["voxel_layer"])
        self.numpy_voxel_layer = VoxelGenerator(**self.config.model["voxel_layer"])
        self.tensor_voxel_layer.training = False
        self.numpy_voxel_layer.training = False
        self.ort_sess = ort.InferenceSession(model_path)

        # build anchor generator
        anchor_generator_config = self.config.model["bbox_head"]["anchor_generator"].copy()
        anchor_generator_config.pop("type")
        self.anchor_generator = AlignedAnchor3DRangeGenerator(**anchor_generator_config)

        # build bbox_coder
        self.bbox_coder = DeltaXYZWLHRBBoxCoder()
        self.box_code_size = self.bbox_coder.code_size

        # config
        self.dir_offset = -np.pi / 2
        self.dir_limit_offset = 0
        self.num_classes = self.config.model["bbox_head"]["num_classes"]

    def detect(self, points):
        voxels, num_points, coors = self._voxelize(points, mode="numpy")
        scores, bbox_preds, dir_scores = self.ort_sess.run(
            None, {"voxels": voxels, "num_points": num_points, "coors": coors}
        )

        # convert to tensor and device is cuda
        scores = torch.from_numpy(scores).cuda()
        bbox_preds = torch.from_numpy(bbox_preds).cuda()
        dir_scores = torch.from_numpy(dir_scores).cuda()
        # append to list
        scores = [scores]
        bbox_preds = [bbox_preds]
        dir_scores = [dir_scores]

        bbox_list = self._get_bbox(scores, bbox_preds, dir_scores)
        bbox_results = [self.bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results
        # # test
        # print(scores.shape, bbox_preds.shape, dir_scores.shape)

    def _voxelize(self, points, mode="tensor"):
        """对输入的点云进行voxel化,使用 hard voxelization

        Args:
            points (np.arrary): 输入点云 [N, M], N代表点云数量,M代表点云维度,一般情况下为3维
        Returns:
            voxel_features ([np.array]): voxel化后的特征
                voxels:[voxel_num,32,4]
                num_points:[voxel_num]
                coors:[voxel_num,4]
        """
        if mode == "tensor":
            voxels, coors, num_points = self.tensor_voxel_layer(points)
        elif mode == "numpy":
            voxels, coors, num_points = self.numpy_voxel_layer.generate(points)

        # coors原始为[voxel_num,4] 在4所在维度， 左边扩充一维度0
        coors = np.concatenate([np.zeros([coors.shape[0], 1]), coors], axis=-1)

        if mode == "numpy":
            coors = coors.astype(np.int32)
        return voxels, num_points, coors

    def _get_bbox(self, cls_scores, bbox_preds, dir_cls_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (torch.Tensor): Multi-level class scores.
            bbox_preds (torch.Tensor): Multi-level bbox predictions.
            dir_scores (torch.Tensor): Multi-level direction class predictions.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)

        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        mlvl_anchors = [anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors]

        result_list = []
        rescale = True
        for img_id in range(1):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            dir_cls_pred_list = [dir_cls_preds[i][img_id].detach() for i in range(num_levels)]

            proposals = self.__get_bboxes_single(
                cls_score_list, bbox_pred_list, dir_cls_pred_list, mlvl_anchors, self.config.model["test_cfg"], rescale
            )
            result_list.append(proposals)
        return result_list

    def __get_bboxes_single(self, cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors, cfg=None, rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            cfg (:obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """

        def xywhr2xyxyr(boxes_xywhr):
            """Convert a rotated boxes in XYWHR format to XYXYR format.

            Args:
                boxes_xywhr (torch.Tensor | np.ndarray): Rotated boxes in XYWHR format.

            Returns:
                (torch.Tensor | np.ndarray): Converted boxes in XYXYR format.
            """
            boxes = torch.zeros_like(boxes_xywhr)
            half_w = boxes_xywhr[..., 2] / 2
            half_h = boxes_xywhr[..., 3] / 2

            boxes[..., 0] = boxes_xywhr[..., 0] - half_w
            boxes[..., 1] = boxes_xywhr[..., 1] - half_h
            boxes[..., 2] = boxes_xywhr[..., 0] + half_w
            boxes[..., 3] = boxes_xywhr[..., 1] + half_h
            boxes[..., 4] = boxes_xywhr[..., 4]
            return boxes

        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
            scores = cls_score.sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)

            nms_pre = cfg["nms_pre"]
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        box_type_3d, box_mode_3d = get_box_type(self.config.data.test.box_type_3d)

        mlvl_bboxes_for_nms = xywhr2xyxyr(box_type_3d(mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        # Add a dummy background class to the front when using sigmoid
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg["score_thr"]
        results = box3d_multiclass_nms(
            mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, score_thr, cfg.max_num, cfg, mlvl_dir_scores
        )
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset, self.dir_limit_offset, np.pi)
            bboxes[..., 6] = dir_rot + self.dir_offset + np.pi * dir_scores.to(bboxes.dtype)
        bboxes = box_type_3d(bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels

    @staticmethod
    def bbox3d2result(bboxes, scores, labels, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape (N, 5).
            labels (torch.Tensor): Labels with shape (N, ).
            scores (torch.Tensor): Scores with shape (N, ).
            attrs (torch.Tensor, optional): Attributes with shape (N, ).
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu())

        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()

        return result_dict


class ROS3DDetector:
    def __init__(self, detector, args):
        self.detector = detector
        self.sub_pc_topic = args.sub_pc_topic
        self.pub_pc_topic = args.pub_pc_topic
        self.pub_detected_object_topic = args.pub_detected_object_topic
        self.args = args

    def start(self):
        rospy.init_node("detection", anonymous=True)
        self.detected_object_publisher = rospy.Publisher(
            self.pub_detected_object_topic, DetectedObjectArray, queue_size=1
        )
        self.pc_publisher = rospy.Publisher(self.pub_pc_topic, PointCloud2, queue_size=1)
        self.pc_subscriber = rospy.Subscriber(self.sub_pc_topic, PointCloud2, self.callback)
        self.__status__()
        rospy.spin()

    def callback(self, pointcloud2):
        rospy.logdebug("debug")

        # timing
        start = time.time()
        points = self.convert_pc2_to_numpy(pointcloud2, False)
        result = self.detector.detect(points)
        end = time.time()
        print("Detection time: {}".format(end - start))

        # # show the results
        # if self.args.fix_frame_id:
        #     pointcloud2.header.frame_id = "lidar"

        detected_object = self.convert_result_to_autoware(result, pointcloud2.header, score_thr=self.args.score_thr)
        self.detected_object_publisher.publish(detected_object)
        self.pc_publisher.publish(pointcloud2)

    def __status__(self):
        print("[ ROS3DDetector ] sub_pc_topic : {}".format(self.sub_pc_topic))
        print("[ ROS3DDetector ] pub_pc_topic : {}".format(self.pub_pc_topic))
        print("[ ROS3DDetector ] pub_detected_object_topic : {}".format(self.pub_detected_object_topic))

    @staticmethod
    def convert_pc2_to_numpy(pc2, if_remove_zeros=False):
        """Convert a PointCloud2 message to a numpy array(Nx4).

        Args:
            pc2 (pointcloud2): pointcloud2 format message.

        Returns:
            numpy.array: Nx4 numpy array.
        """
        pc = pypcd.PointCloud.from_msg(pc2)
        x = pc.pc_data["x"].flatten()
        y = pc.pc_data["y"].flatten()
        z = pc.pc_data["z"].flatten()
        intensity = pc.pc_data["intensity"].flatten()

        # normalize the intensity from 0-255 to 0-1
        intensity = intensity / 255.0

        # convert nan to 0
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        z = np.nan_to_num(z)
        intensity = np.nan_to_num(intensity)

        # if remove zeros points
        if if_remove_zeros:
            mask = np.logical_and(x != 0, y != 0)
            x = x[mask]
            y = y[mask]
            z = z[mask]
            intensity = intensity[mask]

        points = np.zeros((x.shape[0], 4))
        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z
        points[:, 3] = intensity

        # tensor double to float
        points = points.astype(np.float32)
        return points

    @staticmethod
    def convert_result_to_autoware(result, header, score_thr=0.1, class_names=["Pedestrian", "Cyclist", "Car"]):
        """Convert detection results to Autoware format.

        Args:
            result (dict): Detection results.
            score_thr (float): Threshold of score.
            class_names (list): Class names.
            header (Header): Header of the pointcloud2.

            Returns:
                DetectedObjectArray: Autoware detected object array.
        """
        # autoware支持类别=[ "bike", "box" ,"bus", "car", "person" ,"truck" ]
        class_names = ["person", "bike", "car"]

        pred_bboxes = result[0]["boxes_3d"].tensor.numpy()
        pred_scores = result[0]["scores_3d"].numpy()
        pred_labels = result[0]["labels_3d"].numpy()

        if score_thr > 0:
            inds = pred_scores > score_thr
            pred_scores = pred_scores[inds]
            pred_bboxes = pred_bboxes[inds]
            pred_labels = pred_labels[inds]

        detected_object_array = DetectedObjectArray()
        detected_object_array.header = header
        detected_object_array.header.stamp = rospy.Time.now()

        for i in range(len(pred_bboxes)):
            pred_bbox = pred_bboxes[i]
            # 九个数 分别对应 （位置xyz 宽长高 zxy旋转角）
            detected_object = DetectedObject()
            # 记录socre、label
            detected_object.score = pred_scores[i]
            detected_object.label = class_names[pred_labels[i]]

            # valid etc
            detected_object.valid = True
            detected_object.pose_reliable = True
            detected_object.header.frame_id = header.frame_id

            # 位置
            detected_object.pose.position.x = pred_bbox[0]
            detected_object.pose.position.y = pred_bbox[1]
            detected_object.pose.position.z = pred_bbox[2]

            # dimensions xyz对应标准坐标系下的长宽高
            detected_object.dimensions.x = pred_bbox[4]
            detected_object.dimensions.y = pred_bbox[3]
            detected_object.dimensions.z = pred_bbox[5]

            # 高度修正
            detected_object.pose.position.z += detected_object.dimensions.z / 2

            # orientation
            # get quaternion from yaw axis euler angles
            yaw = pred_bbox[6]
            yaw = yaw + np.pi / 2  # fix yaw angle
            r = R.from_euler("z", yaw, degrees=False)
            q = r.as_quat()
            detected_object.pose.orientation.x = q[0]
            detected_object.pose.orientation.y = q[1]
            detected_object.pose.orientation.z = q[2]
            detected_object.pose.orientation.w = q[3]

            detected_object_array.objects.append(detected_object)
        return detected_object_array


# args = parser.parse_args() fun
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--sub_pc_topic", type=str, default="/LIDAR_TOP", help="pointcloud2 topic"
    )  # /os_cloud_node/points /top/rslidar_points
    parser.add_argument("--pub_pc_topic", type=str, default="/LIDAR_TOP/republish", help="pointcloud2 topic")
    parser.add_argument(
        "--pub_detected_object_topic",
        type=str,
        default="/LIDAR_TOP/detected_objects",
        help="detected object topic",
    )
    parser.add_argument("--score_thr", type=float, default=0.5, help="score threshold")
    parser.add_argument("--class_names", type=str, default="Pedestrian,Cyclist,Car", help="class names")
    parser.add_argument("--model_path", type=str, default="../../work_dir/end2end.onnx", help="model path")
    # for debug
    parser.add_argument("--fix_frame_id", type=bool, default=True, help="if use specific frame id lidar")
    parser.add_argument("--remove_zeros", type=bool, default=True, help="if remove_zeros from pointcloud2")
    args = parser.parse_args()
    return args


def main_test():
    args = parse_args()
    data_mode = "numpy"  # numpy or tensor
    bin_path = "/".join((os.getenv("MMDETECTION3D_DIR"), "demo/data/kitti/kitti_000008.bin"))
    config_path = "/".join(
        (os.getenv("MMDETECTION3D_DIR"), "configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py")
    )
    if data_mode == "tensor":
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        points = torch.from_numpy(points).float()
        device = torch.device("cuda:0")
        points = points.to(device)
    elif data_mode == "numpy":
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    config = mmcv.Config.fromfile(config_path)
    pointpillars_detector = PointPillarsDetector(config=config, model_path=args.model_path)
    result = pointpillars_detector.detect(points)

    # publish result
    rospy.init_node("detection", anonymous=True)
    header = Header()
    header.frame_id = "lidar"
    header.stamp = rospy.Time.now()
    detected_object_array = ROS3DDetector.convert_result_to_autoware(result=result, header=header)
    # convert numpy points to pointcloud2
    pc2 = numpy_pc2.array_to_xyzi_pointcloud2f(points, stamp=header.stamp, frame_id=header.frame_id)
    pub_detected_object_array = rospy.Publisher(args.pub_detected_object_topic, DetectedObjectArray, queue_size=1)
    pub_pointcloud2 = rospy.Publisher(args.pub_pc_topic, PointCloud2, queue_size=1)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub_detected_object_array.publish(detected_object_array)
        pub_pointcloud2.publish(pc2)
        rate.sleep()


def main():
    args = parse_args()
    config_path = "/".join(
        (os.getenv("MMDETECTION3D_DIR"), "configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py")
    )
    config = mmcv.Config.fromfile(config_path)

    pointpillars_detector = PointPillarsDetector(config=config, model_path=args.model_path)

    ros_3d_detector = ROS3DDetector(detector=pointpillars_detector, args=args)
    ros_3d_detector.start()


if __name__ == "__main__":
    main()
    # main_test()
