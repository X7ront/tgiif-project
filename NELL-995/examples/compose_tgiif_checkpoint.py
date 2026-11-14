#!/usr/bin/env python3
"""Compose a mature official scorer with a separately trained TGIIF adapter."""

import argparse
import os

import torch


OFFICIAL_PREFIXES = ("sf_lin_layer.", "lin.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_state(path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state dict: {}".format(path))
    return state


def main():
    args = parse_args()
    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(
            "output already exists; pass --force to replace it: {}".format(
                args.output
            )
        )

    adapter_state = load_state(args.adapter_checkpoint)
    official_state = load_state(args.official_checkpoint)
    official_keys = sorted(
        key
        for key in adapter_state
        if key.startswith(OFFICIAL_PREFIXES)
    )
    if not official_keys:
        raise ValueError("adapter checkpoint has no official scorer tensors")

    missing = [key for key in official_keys if key not in official_state]
    mismatched = [
        key
        for key in official_keys
        if key in official_state
        and tuple(adapter_state[key].shape) != tuple(official_state[key].shape)
    ]
    if missing or mismatched:
        raise ValueError(
            "incompatible official checkpoint: missing={}, shape_mismatch={}".format(
                missing, mismatched
            )
        )

    composed = dict(adapter_state)
    for key in official_keys:
        composed[key] = official_state[key].detach().clone()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(composed, args.output)
    print(
        "Composed checkpoint: official_tensors={} total_tensors={} output={}".format(
            len(official_keys), len(composed), args.output
        )
    )


if __name__ == "__main__":
    main()
