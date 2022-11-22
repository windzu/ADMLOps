# Copyright (c) windzu. All rights reserved.

import argparse


class MonoCalibrator:

    def __init__(
        self,
        camera_model,
        pattern,
        rows,
        cols,
        size,
    ):
        self.camera_model = camera_model
        self.pattern = pattern
        self.rows = rows
        self.cols = cols
        self.size = size


def parse_args():
    parser = argparse.ArgumentParser(description='mono calibration')
    parser.add_argument('--camera_model', type=str, default='pinhole')
    parser.add_argument('--pattern', type=str, default='chessboard')
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--cols', type=int, default=6)
    parser.add_argument('--size', type=float, default=0.025)
    parser.add_argument('--device', type=str, default='/dev/video0')
    args = parser.parse_args()
    return args


# so many things to do


def main():
    args = parse_args()

    # check parameters
    # camera_model must be "pinhole" or "fisheye"
    assert args.camera_model in ['pinhole', 'fisheye']
    # pattern type must be "chessboard" or "charuco"
    assert args.pattern in ['chessboard', 'charuco']
    # size must be a float
    assert isinstance(args.size, float)
    # device_name must be a string
    assert isinstance(args.device, str)

    # print args
    print('args:')
    print(f'camera_model: {args.camera_model}')
    print(f'pattern: {args.pattern}')
    print(f'rows: {args.rows}')
    print(f'cols: {args.cols}')
    print(f'size: {args.size}')
    print(f'device: {args.device}')

    # start calibration


if __name__ == '__main__':
    main()
