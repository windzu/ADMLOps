# Copyright (c) windzu. All rights reserved.
# 仅测试使用，尚未开发完毕
import os

import cv2
from mmdeploy_python import Detector

# get mmdeploy model path of faster r-cnn
model_path = '../../work_dir'
# use mmdetection demo image as an input image
image_path = '/'.join((os.getenv('MMDETECTION_DIR'), 'demo/demo.jpg'))

img = cv2.imread(image_path)
detector = Detector(model_path, 'cuda', 0)
bboxes, labels, _ = detector([img])[0]

indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
    [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
    if score < 0.3:
        continue
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)
