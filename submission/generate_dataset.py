#!/usr/bin/env python3
"""Build the ShareGPT JSONL for SmolVLM2 SFT from Bridge, Fractal, Language Table and Ego-100K."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="root with the raw datasets")
    parser.add_argument("--output", type=Path, required=True, help="output dir for annotations.jsonl and frames/")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    raise NotImplementedError(f"generator not implemented yet (input={args.input}, output={args.output})")


if __name__ == "__main__":
    main()
