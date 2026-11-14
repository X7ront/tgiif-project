#!/usr/bin/env python3
"""Prepare common KG benchmarks in the OpenKE *2id.txt format.

The paper "Knowledge Graphs Can be Learned with Just Intersection Features"
uses YAGO3-10, NELL-995, WN18RR, FB15k-237 and FB15k. This utility accepts
an existing raw dataset directory or downloads the standard
DeepGraphLearning/KnowledgeGraphEmbedding text splits, then writes a
self-contained OpenKE benchmark directory.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

DATASETS = {
    "wn18rr": "WN18RR",
    "fb15k237": "FB15k-237",
    "fb15k-237": "FB15k-237",
    "fb15k": "FB15k",
    "yago3-10": "YAGO3-10",
    "yago3_10": "YAGO3-10",
    "nell-995": "NELL-995",
    "nell995": "NELL-995",
}
DOWNLOAD_BASE = (
    "https://raw.githubusercontent.com/DeepGraphLearning/"
    "KnowledgeGraphEmbedding/master/data/{dataset}/{split}.txt"
)
SPLITS = ("train", "valid", "test")
OPENKE_FILES = ("train2id.txt", "valid2id.txt", "test2id.txt",
                "entity2id.txt", "relation2id.txt")


def canonical_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASETS:
        raise ValueError(
            "unknown dataset {!r}; choose from {}".format(
                name, ", ".join(sorted(set(DATASETS.values())))
            )
        )
    return DATASETS[key]


def find_raw_split(source: Path, split: str) -> Path | None:
    for candidate in (
        source / (split + ".txt"),
        source / (split + ".tsv"),
        source / (split + "2id.txt"),
    ):
        if candidate.is_file():
            return candidate
    return None


def parse_raw(path: Path, openke_order: bool = False):
    triples = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace(",", "	").split()
            if len(fields) < 3:
                if lineno == 1 and len(fields) == 1 and fields[0].isdigit():
                    continue
                raise ValueError("{}:{} has fewer than 3 fields".format(path, lineno))
            if openke_order:
                triples.append((fields[0], fields[2], fields[1]))
            else:
                triples.append((fields[0], fields[1], fields[2]))
    if not triples:
        raise ValueError("no triples found in {}".format(path))
    return triples


def write_openke(destination: Path, triples_by_split: dict[str, list[tuple[str, str, str]]]):
    entities = sorted({x for triples in triples_by_split.values() for h, _, t in triples for x in (h, t)})
    relations = sorted({r for triples in triples_by_split.values() for _, r, _ in triples})
    entity_id = {value: i for i, value in enumerate(entities)}
    relation_id = {value: i for i, value in enumerate(relations)}
    destination.mkdir(parents=True, exist_ok=True)

    def write_mapping(path: Path, values):
        with path.open("w", encoding="utf-8") as stream:
            stream.write(str(len(values)) + "\n")
            for value, idx in values.items():
                stream.write("{}\t{}\n".format(value, idx))

    write_mapping(destination / "entity2id.txt", entity_id)
    write_mapping(destination / "relation2id.txt", relation_id)
    for split, triples in triples_by_split.items():
        out = destination / (split + "2id.txt")
        with out.open("w", encoding="utf-8") as stream:
            stream.write(str(len(triples)) + "\n")
            for head, relation, tail in triples:
                stream.write("{} {} {}\n".format(
                    entity_id[head], entity_id[tail], relation_id[relation]
                ))
    print("Prepared {}: {} entities, {} relations, {}".format(
        destination, len(entities), len(relations),
        ", ".join("{}={}".format(k, len(v)) for k, v in triples_by_split.items()),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="WN18RR, FB15k-237, FB15k, YAGO3-10 or NELL-995")
    parser.add_argument("--output-root", default="./benchmarks")
    parser.add_argument("--source-dir", default=None,
                        help="directory containing train/valid/test raw files or *2id files")
    parser.add_argument("--download", action="store_true",
                        help="download missing raw splits from the public repository")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing output directory")
    args = parser.parse_args()

    dataset = canonical_name(args.dataset)
    output = Path(args.output_root).expanduser() / dataset
    if output.exists() and not args.force:
        if all((output / filename).is_file() for filename in OPENKE_FILES):
            print("Already prepared: {}".format(output))
            return
        raise FileExistsError(
            "{} exists but is incomplete; use --force to rebuild".format(output)
        )

    source = Path(args.source_dir).expanduser() if args.source_dir else output / "raw"
    source.mkdir(parents=True, exist_ok=True)
    raw_paths = {}
    for split in SPLITS:
        path = find_raw_split(source, split)
        if path is None and args.download:
            path = source / (split + ".txt")
            url = DOWNLOAD_BASE.format(dataset=dataset, split=split)
            print("Downloading {} -> {}".format(url, path))
            try:
                urllib.request.urlretrieve(url, path)
            except Exception:
                if path.exists():
                    path.unlink()
                raise
        if path is None:
            raise FileNotFoundError(
                "missing {} split in {}; provide --source-dir or --download".format(
                    split, source
                )
            )
        raw_paths[split] = path

    triples = {split: parse_raw(path, path.name.endswith("2id.txt")) for split, path in raw_paths.items()}
    write_openke(output, triples)


if __name__ == "__main__":
    main()