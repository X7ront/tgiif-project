import argparse
import csv
import json
import re
from pathlib import Path


RESULT = re.compile(
    r"Filtered ranking \(valid\): MRR=(?P<mrr>[0-9.]+), "
    r"MR=(?P<mr>[0-9.]+), Hits@10=(?P<h10>[0-9.]+), "
    r"Hits@3=(?P<h3>[0-9.]+), Hits@1=(?P<h1>[0-9.]+)"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.log_dir).glob("*.log")):
        matches = list(RESULT.finditer(path.read_text(
            encoding="utf-8", errors="replace"
        )))
        if not matches:
            continue
        values = {key: float(value) for key, value in matches[-1].groupdict().items()}
        rows.append({
            "run": path.stem,
            "mrr": values["mrr"],
            "mr": values["mr"],
            "hits1": values["h1"],
            "hits3": values["h3"],
            "hits10": values["h10"],
            "hits10_minus_hits1": values["h10"] - values["h1"],
            "strict_hits_gradient": (
                values["h10"] > values["h3"] > values["h1"]
            ),
        })

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(rows[0]) if rows else [
            "run", "mrr", "mr", "hits1", "hits3", "hits10",
            "hits10_minus_hits1", "strict_hits_gradient",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
