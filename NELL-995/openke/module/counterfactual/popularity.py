from pathlib import Path

import numpy as np


def read_openke_triples(path):
    """Read an OpenKE ``*2id.txt`` file as an ``[N, 3]`` int64 array."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        if not first_line:
            raise ValueError("Empty triple file: {}".format(path))
        expected_count = int(first_line)
        triples = np.loadtxt(handle, dtype=np.int64, ndmin=2)

    if triples.shape[1] != 3:
        raise ValueError(
            "Expected three columns (head, tail, relation), got {}".format(
                triples.shape[1]
            )
        )
    if triples.shape[0] != expected_count:
        raise ValueError(
            "Header declares {} triples but {} were read from {}".format(
                expected_count, triples.shape[0], path
            )
        )
    return triples


def build_popularity_treatment(triples, ent_tot, rel_tot):
    """Create a binary, relation-conditioned popularity treatment.

    Entity popularity is measured using train-only endpoint degree. A triple's
    raw popularity is ``log1p(deg(head)) + log1p(deg(tail))``. The binary
    regime is assigned relative to the median inside each relation so that
    counterfactual matching can keep the relation fixed while changing the
    treatment.

    Ties that would leave a relation with only one regime are split by stable
    rank. This fallback is recorded in the returned metadata.
    """
    triples = np.asarray(triples, dtype=np.int64)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("triples must have shape [num_triples, 3]")

    heads, tails, relations = triples.T
    if heads.min() < 0 or tails.min() < 0:
        raise ValueError("Entity ids must be non-negative")
    if relations.min() < 0:
        raise ValueError("Relation ids must be non-negative")
    if heads.max() >= ent_tot or tails.max() >= ent_tot:
        raise ValueError("Entity id exceeds ent_tot")
    if relations.max() >= rel_tot:
        raise ValueError("Relation id exceeds rel_tot")

    entity_degree = np.bincount(
        np.concatenate((heads, tails)), minlength=ent_tot
    ).astype(np.int64)
    relation_frequency = np.bincount(relations, minlength=rel_tot).astype(
        np.int64
    )
    popularity_score = (
        np.log1p(entity_degree[heads]) + np.log1p(entity_degree[tails])
    ).astype(np.float32)

    treatment = np.zeros(len(triples), dtype=np.int64)
    relation_threshold = np.full(rel_tot, np.nan, dtype=np.float32)
    fallback_relations = []

    for relation in np.unique(relations):
        indices = np.flatnonzero(relations == relation)
        scores = popularity_score[indices]
        threshold = float(np.median(scores))
        relation_threshold[relation] = threshold
        labels = (scores > threshold).astype(np.int64)

        if len(indices) >= 2 and np.unique(labels).size < 2:
            order = np.argsort(scores, kind="stable")
            labels = np.zeros(len(indices), dtype=np.int64)
            labels[order[len(indices) // 2:]] = 1
            fallback_relations.append(int(relation))

        treatment[indices] = labels

    return {
        "treatment": treatment,
        "popularity_score": popularity_score,
        "entity_degree": entity_degree,
        "relation_frequency": relation_frequency,
        "relation_threshold": relation_threshold,
        "fallback_relations": fallback_relations,
    }
