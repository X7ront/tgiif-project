import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


CACHE_VERSION = 2


def fingerprint_triples(triples):
    """Return a stable SHA-256 fingerprint for an OpenKE triple array."""
    triples = np.ascontiguousarray(triples, dtype=np.int64)
    return hashlib.sha256(triples.tobytes()).hexdigest()


def _standardize_features(features):
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean, std


def _balanced_neighbor_query(query_features, candidate_features, candidate_indices, max_neighbors):
    neighbor_count = min(int(max_neighbors), len(candidate_indices))
    search = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    search.fit(candidate_features)
    distances, neighbors = search.kneighbors(query_features)

    # A 4D one-hop Triple Sketch creates many exact-distance ties. Standard
    # 1-NN always picks the first row and creates artificial CF hubs. Select
    # the least-used candidate only among candidates tied at the true minimum;
    # matching distance is therefore never relaxed.
    usage = np.zeros(len(candidate_indices), dtype=np.int64)
    selected = np.empty(len(query_features), dtype=np.int64)
    selected_distance = distances[:, 0].astype(np.float32)
    for row in range(len(query_features)):
        tied = np.flatnonzero(
            np.isclose(distances[row], distances[row, 0], rtol=1e-6, atol=1e-8)
        )
        local_candidates = neighbors[row, tied]
        local_usage = usage[local_candidates]
        chosen_local = local_candidates[np.argmin(local_usage)]
        usage[chosen_local] += 1
        selected[row] = candidate_indices[chosen_local]
    return selected, selected_distance


def _nearest_opposite_treatment(
    features, relations, treatment, max_neighbors=64
):
    match_index = np.full(len(features), -1, dtype=np.int64)
    match_distance = np.full(len(features), np.inf, dtype=np.float32)

    for relation in np.unique(relations):
        relation_indices = np.flatnonzero(relations == relation)
        low_indices = relation_indices[treatment[relation_indices] == 0]
        high_indices = relation_indices[treatment[relation_indices] == 1]
        if len(low_indices) == 0 or len(high_indices) == 0:
            continue

        selected, distances = _balanced_neighbor_query(
            features[low_indices],
            features[high_indices],
            high_indices,
            max_neighbors,
        )
        match_index[low_indices] = selected
        match_distance[low_indices] = distances

        selected, distances = _balanced_neighbor_query(
            features[high_indices],
            features[low_indices],
            low_indices,
            max_neighbors,
        )
        match_index[high_indices] = selected
        match_distance[high_indices] = distances

    return match_index, match_distance


def build_counterfactual_cache(
    triples,
    subgraph_features,
    treatment_info,
    max_distance_percentile=95.0,
    match_neighbor_count=64,
):
    """Build train-only, same-relation, opposite-treatment CF matches."""
    triples = np.asarray(triples, dtype=np.int64)
    features = np.asarray(subgraph_features, dtype=np.float32)
    treatment = np.asarray(treatment_info["treatment"], dtype=np.int64)

    if len(triples) != len(features) or len(triples) != len(treatment):
        raise ValueError("triples, features, and treatment must have equal length")
    if not 0.0 < max_distance_percentile <= 100.0:
        raise ValueError("max_distance_percentile must be in (0, 100]")
    if int(match_neighbor_count) < 1:
        raise ValueError("match_neighbor_count must be at least 1")

    normalized, feature_mean, feature_std = _standardize_features(features)
    relations = triples[:, 2]
    match_index, match_distance = _nearest_opposite_treatment(
        normalized,
        relations,
        treatment,
        max_neighbors=match_neighbor_count,
    )

    matched = match_index >= 0
    if not np.any(matched):
        raise ValueError("No opposite-treatment match exists in any relation")
    distance_threshold = float(
        np.percentile(match_distance[matched], max_distance_percentile)
    )
    valid = matched & (match_distance <= distance_threshold)

    cf_triples = np.full_like(triples, -1)
    cf_features = np.zeros_like(features, dtype=np.float32)
    cf_treatment = np.full_like(treatment, -1)
    cf_triples[matched] = triples[match_index[matched]]
    cf_features[matched] = features[match_index[matched]]
    cf_treatment[matched] = treatment[match_index[matched]]

    if np.any(relations[valid] != cf_triples[valid, 2]):
        raise AssertionError("Counterfactual relation mismatch")
    if np.any(treatment[valid] == cf_treatment[valid]):
        raise AssertionError("Counterfactual treatment did not change")

    return {
        "version": CACHE_VERSION,
        "triple_fingerprint": fingerprint_triples(triples),
        "triples": torch.from_numpy(triples),
        "subgraph_feature": torch.from_numpy(features),
        "treatment": torch.from_numpy(treatment),
        "popularity_score": torch.from_numpy(
            np.asarray(treatment_info["popularity_score"], dtype=np.float32)
        ),
        "cf_index": torch.from_numpy(match_index),
        "cf_triples": torch.from_numpy(cf_triples),
        "cf_subgraph_feature": torch.from_numpy(cf_features),
        "cf_treatment": torch.from_numpy(cf_treatment),
        "cf_match_distance": torch.from_numpy(match_distance),
        "cf_valid_mask": torch.from_numpy(valid),
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "max_distance_percentile": float(max_distance_percentile),
        "distance_threshold": distance_threshold,
        "match_neighbor_count": int(match_neighbor_count),
        "valid_count": int(valid.sum()),
        "matched_count": int(matched.sum()),
        "total_count": int(len(triples)),
        "treatment_fallback_relations": list(
            treatment_info.get("fallback_relations", [])
        ),
    }


def load_counterfactual_cache(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Counterfactual cache not found: {}".format(path))
    cache = torch.load(str(path), map_location="cpu")
    if cache.get("version") != CACHE_VERSION:
        raise ValueError(
            "Unsupported counterfactual cache version: {}".format(
                cache.get("version")
            )
        )
    return cache


def validate_counterfactual_cache(
    cache,
    train_triples,
    dataset=None,
    hash_hops=None,
    feature_mode=None,
    feature_dim=None,
):
    """Verify that a cache belongs to the current train split and run."""
    train_triples = np.asarray(train_triples, dtype=np.int64)
    if train_triples.ndim != 2 or train_triples.shape[1] != 3:
        raise ValueError("train_triples must have shape [num_triples, 3]")

    expected_fingerprint = cache.get("triple_fingerprint")
    actual_fingerprint = fingerprint_triples(train_triples)
    if not expected_fingerprint:
        raise ValueError("Counterfactual cache has no train split fingerprint")
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Counterfactual cache train split fingerprint does not match "
            "the current train2id.txt"
        )

    cached_triples = cache.get("triples")
    if cached_triples is None:
        raise ValueError("Counterfactual cache has no embedded train triples")
    if torch.is_tensor(cached_triples):
        cached_triples = cached_triples.cpu().numpy()
    if fingerprint_triples(cached_triples) != expected_fingerprint:
        raise ValueError(
            "Counterfactual cache embedded triples do not match its fingerprint"
        )
    if int(cache.get("total_count", -1)) != len(train_triples):
        raise ValueError("Counterfactual cache train triple count does not match")

    expected_metadata = {
        "dataset": dataset,
        "hash_hops": hash_hops,
        "feature_dim": feature_dim,
    }
    for name, expected in expected_metadata.items():
        if expected is not None and cache.get(name) != expected:
            raise ValueError(
                "Counterfactual cache {} does not match run".format(name)
            )

    if feature_mode is not None:
        cached_feature_mode = cache.get("feature_mode", "triple")
        if cached_feature_mode != feature_mode:
            raise ValueError(
                "Counterfactual cache feature_mode does not match run"
            )
    return actual_fingerprint


class CounterfactualBatchEnricher:
    """Add one offline factual/CF pair per positive sample.

    Negative samples remain untouched. Auxiliary losses therefore operate on
    exactly ``batch_size`` factual/CF pairs and are not multiplied by
    ``neg_ent``.
    """

    def __init__(self, data_loader, cache):
        self.data_loader = data_loader
        self.cache = cache
        self.batch_size = data_loader.get_batch_size()
        triples = cache["triples"].cpu().numpy()
        validate_counterfactual_cache(cache, triples)
        self._index = {
            (int(head), int(relation), int(tail)): index
            for index, (head, tail, relation) in enumerate(triples)
        }

    def _enrich(self, data):
        heads = np.asarray(data["batch_h"][:self.batch_size])
        tails = np.asarray(data["batch_t"][:self.batch_size])
        relations = np.asarray(data["batch_r"][:self.batch_size])
        indices = np.fromiter(
            (
                self._index.get((int(head), int(relation), int(tail)), -1)
                for head, relation, tail in zip(heads, relations, tails)
            ),
            dtype=np.int64,
            count=self.batch_size,
        )

        found = indices >= 0
        safe_indices = indices.copy()
        safe_indices[~found] = 0
        safe_indices_t = torch.from_numpy(safe_indices)
        target_device = data["subgraph_feature"].device

        valid = self.cache["cf_valid_mask"][safe_indices_t].clone()
        valid &= torch.from_numpy(found)
        data["cf_subgraph_feature"] = self.cache["cf_subgraph_feature"][
            safe_indices_t
        ].to(target_device)
        cf_triples = self.cache["cf_triples"][safe_indices_t].to(
            target_device
        )
        data["cf_batch_h"] = cf_triples[:, 0]
        data["cf_batch_t"] = cf_triples[:, 1]
        data["cf_batch_r"] = cf_triples[:, 2]
        data["factual_treatment"] = self.cache["treatment"][
            safe_indices_t
        ].to(target_device, dtype=torch.float32).view(-1, 1)
        data["cf_treatment"] = self.cache["cf_treatment"][safe_indices_t].to(
            target_device, dtype=torch.float32
        ).view(-1, 1)
        data["cf_valid_mask"] = valid.to(target_device)
        data["cf_match_distance"] = self.cache["cf_match_distance"][
            safe_indices_t
        ].to(target_device)
        data["cf_target_index"] = self.cache["cf_index"][safe_indices_t].to(
            target_device
        )
        return data

    def __iter__(self):
        for data in self.data_loader:
            yield self._enrich(data)

    def __len__(self):
        return len(self.data_loader)

    def __getattr__(self, name):
        return getattr(self.data_loader, name)
