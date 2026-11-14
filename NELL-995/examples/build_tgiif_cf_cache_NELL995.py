import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from openke.data import TrainDataLoader
from openke.module.counterfactual import (
    build_counterfactual_cache,
    build_popularity_treatment,
    read_openke_triples,
)
from openke.utils import set_seeds_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a CTS-Net train-only popularity counterfactual cache"
    )
    parser.add_argument("--dataset", default="NELL-995")
    parser.add_argument("--data-path", default="./benchmarks/NELL-995/")
    parser.add_argument("--hash-hops", type=int, default=1)
    parser.add_argument(
        "--feature-mode", choices=("triple", "pairwise"), default="triple",
        help="Feature family used by the CTS base scorer",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--distance-percentile", type=float, default=95.0)
    parser.add_argument("--match-neighbors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to ./counterfactual_cache/NELL-995_popularity_{feature_mode}_h{hops}.pt",
    )
    return parser.parse_args()


@torch.no_grad()
def extract_train_features(train_dataloader, triples, chunk_size):
    feature_batches = []
    device = train_dataloader.device
    ent_tot = train_dataloader.get_ent_tot()
    hashes = train_dataloader.hash_dataset.get_hashes()
    cards = train_dataloader.hash_dataset.get_cards()
    extractor = train_dataloader.hash_dataset.elph_hashes

    for start in range(0, len(triples), chunk_size):
        batch = triples[start:start + chunk_size]
        links = torch.from_numpy(batch[:, [0, 1]]).to(device)
        relationships = torch.from_numpy(batch[:, 2] + ent_tot).to(device)
        if train_dataloader.feature_mode == "pairwise":
            features = extractor.get_pairwise_features(links, hashes, cards)
        else:
            features = extractor.get_subgraph_features(
                links, relationships, hashes, cards
            )
        feature_batches.append(features.detach().cpu().float())
    return torch.cat(feature_batches, dim=0)


def main():
    args = parse_args()
    if not args.dataset or "/" in args.dataset or "\\" in args.dataset:
        raise ValueError("dataset must be a non-empty name without path separators")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seeds_all(args.seed)

    triples = read_openke_triples(
        os.path.join(args.data_path, "train2id.txt")
    )
    train_dataloader = TrainDataLoader(
        in_path=args.data_path,
        nbatches=100,
        threads=args.threads,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=0,
        neg_rel=0,
        device=args.device,
        max_hash_hops=args.hash_hops,
        feature_mode=args.feature_mode,
    )
    if len(triples) != train_dataloader.get_triple_tot():
        raise ValueError("train2id count does not match Base.so train total")

    treatment_info = build_popularity_treatment(
        triples,
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
    )
    features = extract_train_features(
        train_dataloader, triples, args.chunk_size
    )
    cache = build_counterfactual_cache(
        triples,
        features.numpy(),
        treatment_info,
        max_distance_percentile=args.distance_percentile,
        match_neighbor_count=args.match_neighbors,
    )
    cache.update({
        "dataset": args.dataset,
        "hash_hops": args.hash_hops,
        "feature_mode": args.feature_mode,
        "feature_dim": int(features.shape[1]),
        "ent_tot": train_dataloader.get_ent_tot(),
        "rel_tot": train_dataloader.get_rel_tot(),
        "seed": args.seed,
    })

    output = args.output or (
        "./counterfactual_cache/{}_popularity_{}_h{}.pt".format(
            args.dataset, args.feature_mode, args.hash_hops
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(cache, output)

    treatment = cache["treatment"].numpy()
    print("Saved counterfactual cache: {}".format(output))
    print(
        "Treatment balance: low={}, high={}".format(
            int((treatment == 0).sum()), int((treatment == 1).sum())
        )
    )
    print(
        "CF coverage: {}/{} ({:.2%}), distance threshold={:.6f}".format(
            cache["valid_count"],
            cache["total_count"],
            cache["valid_count"] / float(cache["total_count"]),
            cache["distance_threshold"],
        )
    )
    print(
        "Relation tie fallbacks: {}".format(
            cache["treatment_fallback_relations"]
        )
    )


if __name__ == "__main__":
    main()
