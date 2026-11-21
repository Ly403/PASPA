#!/usr/bin/env python3
"""Create a run-local TalkSHOW body VQ or PixelCNN config."""

import argparse
import json
from pathlib import Path


CONFIG_NAMES = {
    "vq": "body_vq.json",
    "pixel": "body_pixel.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(CONFIG_NAMES), required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--print-every", type=int, required=True)
    parser.add_argument("--vq-checkpoint", type=Path)
    args = parser.parse_args()

    positive = (
        args.epochs,
        args.batch_size,
        args.save_every,
        args.print_every,
    )
    if any(value < 1 for value in positive):
        parser.error("epochs/batch-size/save-every/print-every must be positive")
    if args.stage == "pixel" and args.vq_checkpoint is None:
        parser.error("--vq-checkpoint is required for the pixel stage")
    if args.stage == "vq" and args.vq_checkpoint is not None:
        parser.error("--vq-checkpoint is only valid for the pixel stage")

    source = (
        args.upstream_dir.resolve() / "config" / CONFIG_NAMES[args.stage]
    )
    with source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["Data"]["data_root"] = str(args.data_root.resolve())
    config["Data"]["pklname"] = (
        f"_talkshow_body_{args.stage}_{args.cache_tag}.pkl"
    )
    config["Train"]["epochs"] = args.epochs
    config["DataLoader"]["batch_size"] = args.batch_size
    config["Log"]["save_every"] = args.save_every
    config["Log"]["print_every"] = args.print_every

    if args.stage == "pixel":
        checkpoint = args.vq_checkpoint.resolve()
        if not checkpoint.is_file():
            parser.error(f"VQ checkpoint does not exist: {checkpoint}")
        config["Model"]["vq_path"] = str(checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

