# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models import BaseDetector

# local
from mmdet.models import YOLOX


@DETECTORS.register_module()
class YOLOPV2(YOLOX):
    """在YOLOX的基础上,增加了drivable_head和lane_head,实现YOLOPV2的复现

    相校于YOLOX的区别:
    1. 增加了drivable_head和lane_head,这两个head是分割任务的head,用于预测drivable area和lane
    2. YOLOPV2原版本使用的是Anchor-based的方法,这里将其改为Anchor-free的方法

    需要做的工作如下:
    1. 修改YOLOX的forward_train,使其支持多个head
    """

    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        drivable_head=None,  # new head
        lane_head=None,  # new head
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        input_size=(640, 640),
        size_multiplier=32,
        random_size_range=(15, 25),
        random_size_interval=10,
        init_cfg=None,
    ):
        super(YOLOPV2, self).__init__(
            backbone,
            neck,
            bbox_head,
            train_cfg,
            test_cfg,
            pretrained,
            init_cfg,
        )

        # new head init
        self.drivable_head = build_head(drivable_head)
        self.lane_head = build_head(lane_head)

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
    ):
        """
        forward_train需要重写,使其支持如下功能:
        1. 支持多个head,包括bbox类型的head和segmentation类型的head
        2. 支持多个loss

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        img, gt_bboxes = self._preprocess(img, gt_bboxes)

        losses = super(YOLOX, self).forward_train(
            img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize(device=img.device)
        self._progress_in_iter += 1

        return losses

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_drivable_head(self):
        """bool: whether the detector has a drivable head"""
        return (
            hasattr(self, "drivable_head") and self.drivable_head is not None
        )

    @property
    def with_lane_head(self):
        """bool: whether the detector has a lane head"""
        return hasattr(self, "lane_head") and self.lane_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        **kwargs,
    ):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        bbox_losses = self.bbox_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )
        losses.update(bbox_losses)

        drivable_losses = self.drivable_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
        )
        losses.update(drivable_losses)

        lane_losses = self.lane_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
        )
        losses.update(lane_losses)

        return losses

    async def async_simple_test(
        self, img, img_meta, proposals=None, rescale=False
    ):
        """Async test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        x = self.extract_feat(img)

        return await self.bbox_head.async_simple_test(
            x, img_meta, rescale=rescale
        )

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, "Bbox head must be implemented."
        x = self.extract_feat(img)

        return self.bbox_head.simple_test(x, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale
        )

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]["img_shape_for_onnx"] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, "onnx_export"):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} can not "
                f"be exported to ONNX. Please refer to the "
                f"list of supported models,"
                f"https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx"  # noqa E501
            )
