#!/usr/bin/env python3
"""Download the pinned NELL-995 split and convert it to OpenKE format.

The source mirror stores triples as ``head relation tail`` without a count
header.  This project uses OpenKE's ``head tail relation`` layout with the
number of rows on the first line.  Entity and relation dictionaries are
reconstructed losslessly from the source pickle mappings.
"""

import argparse
import hashlib
import json
import pickle
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "Wimmics/semantically-enriched-link-prediction-datasets"
REVISION = "2a1e389dab08f1239bf946c3d83be7600dc15937"
BASE_URL = "https://raw.githubusercontent.com/{}/{}/NELL-995%2B".format(
    REPOSITORY, REVISION
)
SOURCE_FILES = {
    "train2id.txt": (
        "train2id.txt",
        "ebb888d92106643b8eeaf49d711f6b1223c6accbc7e4bf25a32937778617309d",
    ),
    "valid2id.txt": (
        "valid2id.txt",
        "82c76a40abda9f8b134ab5abc563180610305bd9b886f4bd01443ab59fb06584",
    ),
    "test2id.txt": (
        "test2id.txt",
        "f45b190cc06ae16e7cf65b7f26397832a77954037dff61454e968fb9dc73b778",
    ),
    "ent2id.pkl": (
        "pickle/ent2id.pkl",
        "e54e60c0335cc4e85d1899a02930f7d73ca46fba49c3e164473d12f4ea2e05bc",
    ),
    "rel2id.pkl": (
        "pickle/rel2id.pkl",
        "8e8d8cfb2512499966eb99277826e5bfd4422208e437aca53cd8ae91c1fd9d77",
    ),
}
EXPECTED = {
    "train2id.txt": 149678,
    "valid2id.txt": 543,
    "test2id.txt": 3992,
    "entities": 75492,
    "relations": 200,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sources(source_dir):
    source_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for local_name, (relative_url, expected_sha) in SOURCE_FILES.items():
        target = source_dir / local_name
        url = BASE_URL + "/" + urllib.parse.quote(relative_url, safe="/")
        if not target.exists() or sha256(target) != expected_sha:
            print("Downloading {}".format(url))
            urllib.request.urlretrieve(url, target)
        actual_sha = sha256(target)
        if actual_sha != expected_sha:
            raise ValueError(
                "SHA256 mismatch for {}: expected {}, got {}".format(
                    target, expected_sha, actual_sha
                )
            )
        records[local_name] = {
            "url": url,
            "sha256": actual_sha,
            "bytes": target.stat().st_size,
        }
    return records


def load_mapping(path, expected_size):
    with path.open("rb") as handle:
        mapping = pickle.load(handle)
    if not isinstance(mapping, dict) or len(mapping) != expected_size:
        raise ValueError("Unexpected mapping in {}".format(path))
    ids = sorted(int(value) for value in mapping.values())
    if ids != list(range(expected_size)):
        raise ValueError("IDs in {} are not contiguous".format(path))
    return mapping


def write_mapping(mapping, output_path):
    ordered = sorted(mapping.items(), key=lambda item: int(item[1]))
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("{}\n".format(len(ordered)))
        for label, identifier in ordered:
            handle.write("{}\t{}\n".format(label, int(identifier)))


def convert_split(source_path, output_path, expected_rows, ent_tot, rel_tot):
    rows = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    "{}:{} does not contain three IDs".format(
                        source_path, line_number
                    )
                )
            head, relation, tail = map(int, fields)
            if not (0 <= head < ent_tot and 0 <= tail < ent_tot):
                raise ValueError("Entity ID out of range at line {}".format(line_number))
            if not 0 <= relation < rel_tot:
                raise ValueError("Relation ID out of range at line {}".format(line_number))
            rows.append((head, tail, relation))
    if len(rows) != expected_rows:
        raise ValueError(
            "{} has {} rows, expected {}".format(
                source_path, len(rows), expected_rows
            )
        )
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("{}\n".format(len(rows)))
        for head, tail, relation in rows:
            handle.write("{}\t{}\t{}\n".format(head, tail, relation))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and convert the pinned standard NELL-995 split"
    )
    parser.add_argument("--source-dir", default="./data_source/NELL-995")
    parser.add_argument("--output-dir", default="./benchmarks/NELL-995")
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_records = download_sources(source_dir)
    ent2id = load_mapping(source_dir / "ent2id.pkl", EXPECTED["entities"])
    rel2id = load_mapping(source_dir / "rel2id.pkl", EXPECTED["relations"])
    write_mapping(ent2id, output_dir / "entity2id.txt")
    write_mapping(rel2id, output_dir / "relation2id.txt")

    split_sets = {}
    for split_name in ("train2id.txt", "valid2id.txt", "test2id.txt"):
        rows = convert_split(
            source_dir / split_name,
            output_dir / split_name,
            EXPECTED[split_name],
            EXPECTED["entities"],
            EXPECTED["relations"],
        )
        split_sets[split_name] = set(rows)

    overlaps = {
        "train_valid": len(split_sets["train2id.txt"] & split_sets["valid2id.txt"]),
        "train_test": len(split_sets["train2id.txt"] & split_sets["test2id.txt"]),
        "valid_test": len(split_sets["valid2id.txt"] & split_sets["test2id.txt"]),
    }
    if any(overlaps.values()):
        raise ValueError("NELL-995 splits overlap: {}".format(overlaps))

    output_records = {}
    for name in (
        "entity2id.txt",
        "relation2id.txt",
        "train2id.txt",
        "valid2id.txt",
        "test2id.txt",
    ):
        path = output_dir / name
        output_records[name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    provenance = {
        "dataset": "NELL-995",
        "source_repository": "https://github.com/{}".format(REPOSITORY),
        "source_revision": REVISION,
        "source_subdirectory": "NELL-995+",
        "mirror_repository_license": "LGPL-2.1",
        "upstream_split_repository": (
            "https://github.com/otiliastr/coper/tree/master/"
            "CoPER_ConvE/data/nell-995"
        ),
        "source_layout": "head relation tail, no count header",
        "output_layout": "OpenKE head tail relation, first-line count",
        "expected_statistics": EXPECTED,
        "split_overlap_counts": overlaps,
        "source_files": source_records,
        "output_files": output_records,
    }
    provenance_path = output_dir / "DATASET_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Prepared NELL-995 at {}".format(output_dir))
    print(json.dumps(EXPECTED, sort_keys=True))
    print("Split overlap counts: {}".format(overlaps))


if __name__ == "__main__":
    main()
