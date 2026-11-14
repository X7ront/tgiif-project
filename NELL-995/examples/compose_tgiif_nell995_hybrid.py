#!/usr/bin/env python3
"""Combine a mature official scorer with a separately trained TGIIF adapter."""

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch


OFFICIAL_PREFIXES = ("sf_lin_layer.", "lin.")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    if not isinstance(state, (dict, OrderedDict)):
        raise TypeError("Checkpoint does not contain a state dictionary: {}".format(path))
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    official_path = Path(args.official)
    adapter_path = Path(args.adapter)
    output_path = Path(args.output)
    official = load_state(official_path)
    adapter = load_state(adapter_path)

    official_keys = [
        key for key in adapter if key.startswith(OFFICIAL_PREFIXES)
    ]
    if len(official_keys) != 23:
        raise RuntimeError(
            "Expected 23 official scorer tensors, found {}".format(len(official_keys))
        )

    hybrid = OrderedDict()
    for key, adapter_value in adapter.items():
        if key in official_keys:
            if key not in official:
                raise KeyError("Official checkpoint is missing {}".format(key))
            source_value = official[key]
            if source_value.shape != adapter_value.shape:
                raise ValueError(
                    "Shape mismatch for {}: official={} adapter={}".format(
                        key, tuple(source_value.shape), tuple(adapter_value.shape)
                    )
                )
            hybrid[key] = source_value.detach().clone()
        else:
            hybrid[key] = adapter_value.detach().clone()

    for key in official_keys:
        if not torch.equal(hybrid[key], official[key]):
            raise RuntimeError("Official tensor verification failed: {}".format(key))
    for key in adapter:
        if key not in official_keys and not torch.equal(hybrid[key], adapter[key]):
            raise RuntimeError("Adapter tensor verification failed: {}".format(key))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hybrid, output_path)
    report = {
        "official_checkpoint": str(official_path),
        "official_sha256": sha256(official_path),
        "adapter_checkpoint": str(adapter_path),
        "adapter_sha256": sha256(adapter_path),
        "output_checkpoint": str(output_path),
        "output_sha256": sha256(output_path),
        "official_tensor_count": len(official_keys),
        "adapter_tensor_count": len(adapter) - len(official_keys),
        "official_prefixes": list(OFFICIAL_PREFIXES),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".composition.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
