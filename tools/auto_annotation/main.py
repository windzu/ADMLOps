# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

# local
from auto_annotation import AutoAnnotation


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input file path or dir path")
    parser.add_argument("--type", default="scalabel", choices=["scalabel", "voc"], help="target format")
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score-thr", type=float, default=0.3, help="bbox score threshold")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    auto_annotation = AutoAnnotation(
        input=args.input,
        type=args.type,
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        score_thr=args.score_thr,
    )
    auto_annotation.run()


if __name__ == "__main__":
    main()
