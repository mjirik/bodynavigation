#! /usr/bin/env python
"""Run BodyNavigation for one CT image or one directory containing it."""

import argparse
from pathlib import Path

from batch_organ_detection import processData


def _default_name(input_path):
    path = Path(input_path)
    if path.is_dir():
        return path.name
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def main():
    parser = argparse.ArgumentParser(
        description="Run BodyNavigation for one CT image or case directory."
    )
    parser.add_argument("-i", "--input", required=True, help="CT image or case directory")
    parser.add_argument("-o", "--output", required=True, help="directory for PNG and traceback outputs")
    parser.add_argument(
        "-p",
        "--parts",
        default="bones_stats,vessels,vessels_stats",
        help="comma-separated parts to process; use None to disable all parts",
    )
    parser.add_argument("--name", help="output basename; defaults to the input name")
    parser.add_argument("--dump", help="directory for preprocessed data")
    parser.add_argument("--draw-depth", action="store_true", help="use solid depth rendering")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error("input does not exist: %s" % input_path)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    parts = [] if args.parts.strip().lower() == "none" else [
        part.strip().lower() for part in args.parts.split(",")
    ]
    processData(
        str(input_path),
        args.name or _default_name(input_path),
        str(output_path),
        parts=parts,
        dumpdir=args.dump,
        draw_depth=args.draw_depth,
    )


if __name__ == "__main__":
    main()
