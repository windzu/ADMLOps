import os
import tensorrt as trt
import pycuda.driver as cuda
from argparse import ArgumentParser
import time

# import pycuda.driver as cuda2
import pycuda.autoinit
import numpy as np
import cv2


# mmlab
from mmdeploy_python import Detector


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file path")
    parser.add_argument("engine", default="./end2end.engine", help="engine file path")
    parser.add_argument("--test_img", default="default", help="img path")
    parser.add_argument("--score_thr", type=float, default=0.1, help="bbox score threshold")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # init detector
    assert os.path.exists(args.engine), "engine file not found"
    detector = Detector(args.engine, "cuda", 0)

    # load img
    assert os.path.exists(args.test_img), "img file not found"
    img = cv2.imread(args.test_img)
    batch_size = 8
    imgs = []
    for i in range(batch_size):
        imgs.append(img)

    # detect
    start_time = time.time()
    bboxes, labels, _ = detector.batch(imgs)[0]
    print("batch_size is {}, cost time is {} s".format(batch_size, time.time() - start_time))


#     # postprocess
#     indices = [i for i in range(len(bboxes))]
#     for index, bbox, label_id in zip(indices, bboxes, labels):
#         [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
#         if score < 0.3:
#             continue
#         cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))
#
#     cv2.imwrite("output_detection.png", img)


if __name__ == "__main__":
    main()
