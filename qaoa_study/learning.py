"""Training-only fixed starts under the common unweighted MaxCut symmetries."""

import numpy as np


_ANGLE_TIE = 32 * np.finfo(np.float64).eps * np.pi
_DISTANCE_TIE = 1e-12
B2_RULES = {
    "version": "b2-common-quotient-v2",
    "gamma_period": 2 * np.pi,
    "beta_period": np.pi / 2,
    "canonical_interval": "[-T/2,T/2)",
    "half_period_boundary": "within 8*float64_eps*T maps to -T/2",
    "sign_rule": "one global +/-; smaller value at the largest-separation coordinate",
    "angle_tie_atol": _ANGLE_TIE,
    "separation_tie": "first coordinate within angle_tie_atol of largest gap; exact lexicographic if all gaps are tiny",
    "distance": "Euclidean radians without period normalization; minimum over global +/-",
    "medoid_objective": "sum of quotient distances, one vote per isomorphism class",
    "distance_tie_atol": _DISTANCE_TIE,
    "medoid_tie": "smallest iso_class_id within distance_tie_atol of minimum sum",
    "median": "one nearest lift per candidate to frozen medoid; coordinate median; canonicalize once",
    "median_even_count": "arithmetic mean of the two middle coordinates",
}


def _periods(p):
    if p not in (1, 2):
        raise ValueError("B2 supports p=1 or p=2 only.")
    return np.array([2 * np.pi] * p + [np.pi / 2] * p)


def _angles(theta):
    value = np.asarray(theta, dtype=np.float64)
    if value.ndim != 1 or value.size not in (2, 4) or not np.isfinite(value).all():
        raise ValueError("Expected 2 or 4 finite angles in [gamma..., beta...] order.")
    return value


def _wrap(theta, periods):
    # fmod preserves small signed angles; adding T/2 before modulo would lose them.
    wrapped = np.fmod(theta, periods)
    half = periods / 2
    wrapped = np.where(wrapped >= half, wrapped - periods, wrapped)
    wrapped = np.where(wrapped < -half, wrapped + periods, wrapped)
    boundary = np.abs(np.abs(wrapped) - half) <= 8 * np.finfo(np.float64).eps * periods
    return np.where(boundary, -half, wrapped)


def _separated_first(left, right):
    """A tiny coordinate cannot decide between macroscopically different lifts."""
    gaps = np.abs(left - right)
    largest = gaps.max()
    if largest <= _ANGLE_TIE:
        return left if tuple(left) <= tuple(right) else right
    index = int(np.flatnonzero(gaps >= largest - _ANGLE_TIE)[0])
    return left if left[index] < right[index] else right


def canonical_angles(theta):
    """Choose a deterministic common-symmetry representative, up to roundoff.

    Gamma has period 2*pi and beta pi/2, independently by layer. Only a
    simultaneous sign reversal of ALL angles is used. Graph-specific gamma
    reductions are deliberately absent. Boundary/tie rules are in B2_RULES;
    small nonzero angles are never rounded or set to zero.
    """
    theta = _angles(theta)
    periods = _periods(theta.size // 2)
    return _separated_first(_wrap(theta, periods), _wrap(-theta, periods)).copy()


def quotient_distances(angles, p):
    """Return pairwise common-quotient Euclidean distances in radians."""
    periods = _periods(p)
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] != 2 * p or not np.isfinite(angles).all():
        raise ValueError("Expected a finite candidate-by-2p angle matrix.")
    direct = np.linalg.norm(_wrap(angles[:, None] - angles[None, :], periods), axis=-1)
    reversed_sign = np.linalg.norm(_wrap(angles[:, None] + angles[None, :], periods), axis=-1)
    return np.minimum(direct, reversed_sign)


def align_to_anchor(theta, anchor):
    """Return the nearest periodic/global-sign lift to a fixed anchor.

    Both global signs compete by Euclidean distance. Numerical distance ties
    choose the smaller lift at their largest-separation coordinate, independent
    of the original input sign. Each half-period displacement takes its negative
    boundary. The returned lift need not lie in the canonical interval.
    """
    theta, anchor = canonical_angles(theta), _angles(anchor)
    if theta.shape != anchor.shape:
        raise ValueError("Candidate and anchor must have the same depth.")
    periods = _periods(theta.size // 2)
    left = anchor + _wrap(theta - anchor, periods)
    right = anchor + _wrap(-theta - anchor, periods)
    left_distance, right_distance = np.linalg.norm(left - anchor), np.linalg.norm(right - anchor)
    if abs(left_distance - right_distance) <= _DISTANCE_TIE:
        return _separated_first(left, right).copy()
    return left if left_distance < right_distance else right


def fixed_angles(candidates):
    """Fit one medoid and its one-pass aligned median from one label per graph.

    Each candidate supplies iso_class_id and theta. The caller must restrict
    these to complete training references for one depth/fold; this function
    never examines objective values. Raw donor angles remain available for
    provenance. Returned values are JSON-compatible.
    """
    candidates = sorted(candidates, key=lambda row: row["iso_class_id"])
    identities = [row["iso_class_id"] for row in candidates]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("B2 requires a nonempty set with one candidate per isomorphism class.")
    angles = np.stack([canonical_angles(row["theta"]) for row in candidates])
    p = angles.shape[1] // 2
    sums = quotient_distances(angles, p).sum(axis=1)
    index = int(np.flatnonzero(sums <= sums.min() + _DISTANCE_TIE)[0])
    medoid = angles[index]
    lifts = np.stack([align_to_anchor(theta, medoid) for theta in angles])
    median = np.median(lifts, axis=0)
    return {
        "p": p, "candidate_count": len(candidates), "training_graph_ids": identities,
        "medoid": {"iso_class_id": identities[index], "theta": medoid.tolist(),
                   "raw_theta": _angles(candidates[index]["theta"]).tolist(),
                   "distance_sum": float(sums[index])},
        "median": {"theta": canonical_angles(median).tolist(), "unwrapped_theta": median.tolist()},
    }


def select_b2_graphs(library, regime, fold, split):
    """Select Tier2 graph records without changing the frozen global split.

    LOFO excludes all multi-family classes from both sides. Its training side
    uses only global training graphs outside the held-out family; evaluation
    uses only that family's global evaluation graphs. An empty result is
    returned for the fitting/planning caller to reject explicitly.
    """
    if split not in ("training", "evaluation") or regime not in ("random", "lofo"):
        raise ValueError("Expected random/lofo regime and training/evaluation split.")
    if (regime == "random" and fold is not None) or (
            regime == "lofo" and fold not in ("regular", "er", "ba", "sbm")):
        raise ValueError("Random split has no fold; LOFO requires a known held-out family.")
    selected = []
    for graph in library["graphs"]:
        if graph["tier"] != 2 or graph["graph_split"] != split:
            continue
        if regime == "lofo":
            families = graph["families"]
            if not graph["lofo_eligible"] or len(families) != 1:
                continue
            held_out = families[0] == fold
            if held_out != (split == "evaluation"):
                continue
        selected.append(graph)
    return sorted(selected, key=lambda row: row["iso_class_id"])
