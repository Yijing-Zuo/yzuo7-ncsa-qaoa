"""Training-only fixed starts and graph-only theoretical initializations."""

from copy import deepcopy
import hashlib
import json
import math
from time import perf_counter

import networkx as nx
import numpy as np

from qaoa_study.graphs import graph_from_record, validate_graph


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
B3_RULES = {
    "1": {
        "rule_id": "max-degree-gamma-analytic-beta-v1",
        "sources": [{"paper": "https://arxiv.org/html/1706.02998v2#S3",
                     "location": "Theorem 1 and triangle-free regular specialization"}],
        "derivation": "maximum-degree tree gamma; exact beta maximizer of the p1 formula",
        "coefficient_order": "sorted (min(deg(u)-1,deg(v)-1),max(deg(u)-1,deg(v)-1),common_neighbors); math.fsum",
    },
    "2": {
        "rule_id": "mean-degree-arctan-infinite-angle-p2-v1",
        "raw_gamma": [0.3817, 0.6655], "gamma_multiplier": 2.0,
        "beta": [0.4960, 0.2690],
        "sources": [
            {"paper": "https://arxiv.org/html/2110.14206v3#A3",
             "location": "Eq. (2.3), Appendix C Table 4, q=2 p=2",
             "code": "https://github.com/benjaminvillalonga/large-girth-maxcut-qaoa/blob/b5bbc23ad4309af12a95666862cd082bddcac0e3/data.csv",
             "commit": "b5bbc23ad4309af12a95666862cd082bddcac0e3",
             "sha256": "39b1b0d82d4f8040b68b9e0d1870e0edd4e8e6b3529b7216e1ca4cb173ba8030"},
            {"paper": "https://arxiv.org/html/2305.15201v3#S6.SS1",
             "location": "Section 6.1 Eq. (87), unit weights and actual mean degree"}],
        "convention": "H_author=-sum(ZZ)/sqrt(D); C_project=(m-sum(ZZ))/2; beta unchanged",
        "precision": "printed four-decimal constants define this rule exactly",
        "external_optimization_cost": "not measured in this project",
    },
}


def b3_initialization(graph, p):
    """Return a deterministic JSON-compatible initial point using only topology.

    p1 fixes gamma by maximum degree and analytically maximizes beta, without
    searching gamma. p2 rescales external infinite-degree constants by actual
    mean degree. Both are finite-graph heuristics; neither promises a better
    optimized endpoint. Preparation timing belongs to the artifact writer.
    """
    if type(p) is not int or p not in (1, 2):
        raise ValueError("B3 supports integer p=1 or p=2 only.")
    validate_graph(graph)
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    if not m or not nx.is_connected(graph):
        raise ValueError("B3 requires a connected graph with at least one edge.")
    rule = deepcopy(B3_RULES[str(p)])
    diagnostics = {"n": n, "m": m}
    counters = {"analytic_coefficient_passes": int(p == 1), "gamma_search_points": 0,
                "qnode_calls": 0, "degree_scaling_evaluations": int(p == 2)}
    if p == 1:
        degree = dict(graph.degree())
        maximum = max(degree.values())
        gamma = math.atan2(1.0, math.sqrt(maximum - 1))
        cosine, cosine2 = math.cos(gamma), math.cos(2 * gamma)
        triples = []
        for u, v in graph.edges():
            a, b = sorted((degree[u] - 1, degree[v] - 1))
            triples.append((a, b, len(graph[u].keys() & graph[v].keys())))
        triples.sort()
        a_coefficient = math.sin(gamma) / 4 * math.fsum(
            cosine**a + cosine**b for a, b, _ in triples)
        triangle_coefficient = math.fsum(
            cosine**(a + b - 2 * t) * (1 - cosine2**t) for a, b, t in triples) / 4
        beta = math.atan2(2 * a_coefficient, triangle_coefficient) / 4
        radius = math.hypot(a_coefficient, triangle_coefficient / 2)
        theta = [gamma, beta]
        diagnostics.update({
            "max_degree": maximum, "gamma": gamma, "A": a_coefficient,
            "T": triangle_coefficient, "beta": beta,
            "analytic_initial_cut": m / 2 - triangle_coefficient / 2 + radius,
            "tree_angle_cut": m / 2 - triangle_coefficient / 2 + a_coefficient,
            "beta_gain": radius - a_coefficient,
        })
    else:
        mean_degree = 2 * m / n
        scale = math.atan2(1.0, math.sqrt(mean_degree - 1))
        theta = [rule["gamma_multiplier"] * value * scale for value in rule["raw_gamma"]]
        theta += rule["beta"]
        diagnostics.update({"mean_degree": mean_degree, "gamma_scale": scale})
    return {"rule_id": rule["rule_id"], "theta": theta, "diagnostics": diagnostics,
            "provenance": rule, "counters": counters}


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


B4_MODEL_VERSION = "b4-ridge-v1"
B4_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.)
B4_RULES = {
    "label": "B2-v2 inner-training medoid; exact parity orbit nearest radian lift",
    "encoding": "gamma (sin,cos); beta (sin4,cos4); coordinate-major",
    "tie": "distance <= minimum+1e-12; largest-spread-coordinate minimum; tiny exact lexicographic",
    "decode": "any rho<1e-8: whole medoid; otherwise atan2/omega; common B2 canonicalization once",
    "angle_cv": "mean squared radians over graph-valid parity/period/global-sign quotient",
    "success_cv": "graph MSE after clipping to [0,1]; observed successes/50",
    "ridge": "float64 sklearn Ridge(solver=svd,fit_intercept=True); alpha=n_fit*lambda",
    "preprocessing": "training assortativity median; population std; constant <=1e-12*max(1,abs(mean))",
    "cv_seed": 20260921,
    "lambda_tie": "largest lambda within 1e-12 of minimum mean OOF loss",
    "shuffle": "whole training X rows only; SHA256(seed,scope,fold); independent of depth/head",
}


def b4_angle_orbit(theta, parity):
    """Exact MaxCut symmetries; odd-degree shifts reverse beta from that layer.

    The identity exp(-i*pi*C)=product_v Z_v**degree(v) supplies the parity
    generators. Mixed-degree parity has only the common global reversal.
    Each JSON row records the generating mask/sign and its diagonal Jacobian.
    """
    theta = _angles(theta)
    if parity not in ("even", "odd", "mixed"):
        raise ValueError("Unknown graph degree parity.")
    p = theta.size // 2
    result = []
    for mask in range(1 if parity == "mixed" else 2**p):
        transformed, jacobian = theta.copy(), np.ones(2 * p)
        for layer in range(p):
            if mask & (1 << layer):
                transformed[layer] += np.pi
                if parity == "odd":
                    transformed[p + layer:] *= -1
                    jacobian[p + layer:] *= -1
        for sign in (1, -1):
            result.append({"theta": (sign * transformed).tolist(), "mask": mask,
                           "sign": sign, "jacobian": (sign * jacobian).tolist()})
    return result


def b4_align_label(theta, anchor, parity):
    """Choose a nearest orbit lift without undoing the anchor's sign choice."""
    theta, anchor = _angles(theta), _angles(anchor)
    if theta.shape != anchor.shape:
        raise ValueError("Reference and anchor depths differ.")
    orbit = b4_angle_orbit(theta, parity)
    lifts = np.array([anchor + _wrap(np.array(item["theta"]) - anchor,
                                   _periods(theta.size // 2)) for item in orbit])
    distances = np.linalg.norm(lifts - anchor, axis=1)
    active = np.flatnonzero(distances <= distances.min() + _DISTANCE_TIE)
    tied = len(active) > 1
    while len(active) > 1:
        spread = np.ptp(lifts[active], axis=0)
        if spread.max() <= _ANGLE_TIE:
            active = [min(active, key=lambda i: tuple(lifts[i]))]
            break
        coordinate = int(np.flatnonzero((spread > _ANGLE_TIE) &
                         (spread >= spread.max() - _ANGLE_TIE))[0])
        values = lifts[active, coordinate]
        active = active[values <= values.min() + _ANGLE_TIE]
    index = int(active[0])
    return {"raw_theta": theta.tolist(), "degree_parity": parity,
            "mask": orbit[index]["mask"], "sign": orbit[index]["sign"],
            "lift": lifts[index].tolist(), "distance": float(distances[index]), "tie": tied}


def b4_encode_angles(theta):
    """Encode 2p angles into 4p unit-circle columns, preserving the label lift."""
    theta = _angles(theta)
    phase = theta * np.array([1.] * (theta.size // 2) + [4.] * (theta.size // 2))
    return np.column_stack((np.sin(phase), np.cos(phase))).ravel()


def b4_decode_angles(prediction, anchor):
    """Decode a numerical model output; any small circle uses the whole medoid."""
    anchor = _angles(anchor)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (2 * anchor.size,) or not np.isfinite(prediction).all():
        raise ValueError("B4 angle predictions require 4p finite outputs.")
    pairs = prediction.reshape(-1, 2)
    rho = np.hypot(pairs[:, 0], pairs[:, 1])
    if not np.isfinite(rho).all():
        raise ValueError("B4 predicted circle norms must be finite.")
    trigger = np.flatnonzero(rho < 1e-8).tolist()
    theta = anchor if trigger else np.arctan2(pairs[:, 0], pairs[:, 1]) / np.array(
        [1.] * (anchor.size // 2) + [4.] * (anchor.size // 2))
    return {"theta0": canonical_angles(theta).tolist(), "raw_prediction": prediction.tolist(),
            "rho": rho.tolist(), "fallback": bool(trigger), "trigger_coordinates": trigger}


def b4_angle_loss(prediction, target, parity):
    """Graph-applicable quotient squared radian distance, averaged over 2p."""
    prediction, target = _angles(prediction), _angles(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target depths differ.")
    residuals = np.array([_wrap(prediction - np.array(row["theta"]),
                              _periods(target.size // 2)) for row in b4_angle_orbit(target, parity)])
    return float(np.min(np.mean(residuals**2, axis=1)))


def _b4_preprocessor(matrix, names):
    """Fit only the declared missing column; NaN here denotes validated None."""
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    index = names.index("degree_assortativity")
    missing = np.isnan(matrix[:, index])
    median = float(np.median(matrix[~missing, index])) if (~missing).any() else 0.
    matrix[missing, index] = median
    if not np.isfinite(matrix).all():
        raise ValueError("Unexpected nonfinite feature outside declared assortativity missingness.")
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    constant = std <= 1e-12 * np.maximum(1., np.abs(mean))
    return {"assortativity_median": median, "mean": mean.tolist(), "std": std.tolist(),
            "constant_mask": constant.tolist()}


def _b4_transform(matrix, preprocessing, names):
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    index = names.index("degree_assortativity")
    matrix[np.isnan(matrix[:, index]), index] = preprocessing["assortativity_median"]
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite feature encountered during B4 transformation.")
    constant = np.array(preprocessing["constant_mask"], dtype=bool)
    matrix = (matrix - preprocessing["mean"]) / np.where(constant, 1., preprocessing["std"])
    matrix[:, constant] = 0.
    return matrix


def _b4_ridge(matrix, target, regularization):
    """Native SVD Ridge, with alpha scaled for the mean-square objective."""
    from sklearn.linear_model import Ridge

    fit = Ridge(alpha=len(matrix) * regularization, fit_intercept=True, solver="svd")
    fit.fit(np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64))
    return {"coef": np.atleast_2d(fit.coef_).tolist(),
            "intercept": np.atleast_1d(fit.intercept_).tolist()}


def _b4_permutation(ids, seed, scope, fold):
    """Depth/head-independent actual mapping, in sorted training-ID order."""
    if seed is None:
        return np.arange(len(ids))
    key = json.dumps([seed, scope, fold], separators=(",", ":"))
    entropy = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:16], "little")
    return np.random.default_rng(entropy).permutation(len(ids))


def _b4_settings(settings):
    settings = {} if settings is None else dict(settings)
    if set(settings) - {"development", "random_n_splits"}:
        raise ValueError("Unknown B4 settings; the production mathematical contract is fixed.")
    development = settings.get("development", False)
    count = settings.get("random_n_splits", 5)
    if type(development) is not bool or type(count) is not int or count < 2:
        raise ValueError("Invalid B4 development/CV settings.")
    if not development and count != 5:
        raise ValueError("Production B4 random CV requires five folds.")
    return {"development": development, "random_n_splits": count}


def _b4_inner_folds(rows, regime, settings):
    if regime == "random":
        from sklearn.model_selection import StratifiedKFold

        cells = [row["representative_cell"] for row in rows]
        count = settings["random_n_splits"]
        if min(cells.count(cell) for cell in set(cells)) < count:
            raise ValueError("Every representative cell must support all random CV folds.")
        splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=B4_RULES["cv_seed"])
        return [(f"random-{i}", train, valid) for i, (train, valid) in enumerate(
            splitter.split(np.zeros(len(rows)), cells))]
    families = np.array([row["families"][0] for row in rows])
    if len(set(families)) != 3:
        raise ValueError("LOFO inner CV requires exactly the three remaining families.")
    return [(str(family), np.flatnonzero(families != family), np.flatnonzero(families == family))
            for family in sorted(set(families))]


def _b4_index(rows, ids, description):
    keyed = {row["iso_class_id"]: row for row in rows}
    if len(keyed) != len(rows) or set(keyed) != set(ids):
        raise ValueError(f"B4 {description} must cover exactly every declared training graph once.")
    return [keyed[identity] for identity in ids]


def fit_b4_model(graph_rows, feature_rows, reference_candidates, success_labels, *,
                 p, group, regime, fold=None, head="angles", shuffle_seed=None, settings=None):
    """Fit a JSON numerical model on an already selected complete training scope.

    Graph rows use the frozen library schema. Feature rows use b4_feature_row
    plus iso_class_id. References contain iso_class_id, theta and optionally
    source provenance. Success labels contain iso_class_id, successes, trials=50.
    The I/O caller audits reference completeness and binds source/library hashes;
    this function rejects partial coverage, evaluation rows and illegal scopes.
    No objective or optimization call is made here. Development may use fewer
    random CV folds; production settings remain five random or three LOFO folds.
    """
    from .features import B4_FEATURE_VERSION, b4_feature_matrix, b4_feature_names
    import sklearn

    start = perf_counter()
    settings = _b4_settings(settings)
    if type(p) is not int or p not in (1, 2) or head not in ("angles", "success"):
        raise ValueError("B4 supports p=1/2 and angles/success heads.")
    rows = sorted(graph_rows, key=lambda row: row["iso_class_id"])
    ids = [row["iso_class_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("B4 requires unique nonempty graph IDs.")
    selected = select_b2_graphs({"graphs": rows}, regime, fold, "training")
    if [row["iso_class_id"] for row in selected] != ids:
        raise ValueError("B4 input contains evaluation or ineligible LOFO graphs.")
    if shuffle_seed is not None and (type(shuffle_seed) is not int or
            shuffle_seed not in (20260922, 20260923, 20260924) or regime != "random" or group != "F"):
        raise ValueError("B4 null is limited to the three declared random/F seeds.")
    features = _b4_index(feature_rows, ids, "features")
    candidates = _b4_index(reference_candidates, ids, "reference candidates")
    for candidate in candidates:
        if _angles(candidate["theta"]).shape != (2 * p,):
            raise ValueError("B4 reference depth mismatch.")
    parities = []
    for row, feature in zip(rows, features):
        graph = graph_from_record(row)
        validate_graph(graph)
        if not 4 <= len(graph) <= 24 or not nx.is_connected(graph):
            raise ValueError("B4 requires connected simple graphs with 4<=n<=24.")
        parity = {degree % 2 for _, degree in graph.degree()}
        parity = "even" if parity == {0} else "odd" if parity == {1} else "mixed"
        if feature["degree_parity"] != parity or feature["values"]["n"] != len(graph):
            raise ValueError("B4 feature topology/parity disagrees with its graph.")
        parities.append(parity)
    names = list(b4_feature_names(group))
    matrix = b4_feature_matrix(features, group)
    rates = None
    if head == "success":
        labels = _b4_index(success_labels, ids, "success labels")
        if any(type(row["successes"]) is not int or type(row["trials"]) is not int or
               row["trials"] != 50 or not 0 <= row["successes"] <= 50 for row in labels):
            raise ValueError("Success labels require integer successes and exactly 50 trials.")
        rates = np.array([row["successes"] / 50 for row in labels])[:, None]
    scope = regime if fold is None else f"{regime}:{fold}"

    def prepare(indices, marker):
        train_ids = [ids[i] for i in indices]
        preprocessing = _b4_preprocessor(matrix[indices], names)
        transformed = _b4_transform(matrix[indices], preprocessing, names)
        permutation = _b4_permutation(train_ids, shuffle_seed, scope, marker)
        part = {"training_graph_ids": train_ids, "preprocessing": preprocessing,
                "permutation": permutation.tolist(),
                "permuted_feature_graph_ids": [train_ids[i] for i in permutation],
                "seen_joint_types": np.any(np.array([features[i]["joint_counts"] for i in indices]) > 0,
                                            axis=0).tolist()}
        if head == "angles":
            anchor = fixed_angles([candidates[i] for i in indices])["medoid"]
            alignment = [{"iso_class_id": ids[i], **b4_align_label(
                candidates[i]["theta"], anchor["theta"], parities[i])} for i in indices]
            target = np.array([b4_encode_angles(row["lift"]) for row in alignment])
            part.update(anchor=anchor, label_alignment=alignment)
        else:
            target = rates[indices]
        return part, transformed[permutation], target

    losses = np.zeros((len(B4_LAMBDAS), len(rows)))
    folds, diagnostics = [], [[] for _ in B4_LAMBDAS]
    cv_start = perf_counter()
    for marker, train, valid in _b4_inner_folds(rows, regime, settings):
        part, train_matrix, target = prepare(train, marker)
        validation_matrix = _b4_transform(matrix[valid], part["preprocessing"], names)
        part.update(fold=marker, validation_graph_ids=[ids[i] for i in valid])
        part["lambda_scores"] = []
        for k, regularization in enumerate(B4_LAMBDAS):
            ridge = _b4_ridge(train_matrix, target, regularization)
            raw = validation_matrix @ np.array(ridge["coef"]).T + ridge["intercept"]
            for index, prediction in zip(valid, raw):
                if head == "angles":
                    decoded = b4_decode_angles(prediction, part["anchor"]["theta"])
                    losses[k, index] = b4_angle_loss(decoded["theta0"], candidates[index]["theta"],
                                                   parities[index])
                    diagnostics[k].append({"iso_class_id": ids[index], **decoded})
                else:
                    value = float(np.clip(prediction[0], 0., 1.))
                    losses[k, index] = (value - rates[index, 0])**2
                    diagnostics[k].append({"iso_class_id": ids[index], "success_probability": value,
                                           "raw_prediction": float(prediction[0])})
            part["lambda_scores"].append({"lambda": regularization,
                "mean_loss": float(losses[k, valid].mean())})
        folds.append(part)
    scores = losses.mean(axis=1)
    eligible = np.flatnonzero(scores <= scores.min() + _DISTANCE_TIE)
    chosen = int(eligible[-1])
    cv_seconds = perf_counter() - cv_start
    final_start = perf_counter()
    final, train_matrix, target = prepare(np.arange(len(rows)), "final-fit")
    final.update(_b4_ridge(train_matrix, target, B4_LAMBDAS[chosen]))
    final_fit_seconds = perf_counter() - final_start
    model = {"version": B4_MODEL_VERSION, "rules": deepcopy(B4_RULES), "feature_version": B4_FEATURE_VERSION,
             "feature_names": names, "p": p, "group": group, "regime": regime, "fold": fold,
             "head": head, "shuffle_seed": shuffle_seed, "settings": settings, "training_graph_ids": ids,
             "selected_lambda": B4_LAMBDAS[chosen], "model": final,
             "cv": {"folds": folds, "lambda_scores": [{"lambda": value, "mean_loss": float(scores[k]),
                     "loss_by_graph": losses[k].tolist()} for k, value in enumerate(B4_LAMBDAS)],
                    "lambda_tie": len(eligible) > 1, "loss_graph_ids": ids,
                    "selected_oof_predictions": sorted(diagnostics[chosen], key=lambda row: row["iso_class_id"])},
             "reference_candidates": deepcopy(candidates),
             "learning_environment": {"numpy": np.__version__, "sklearn": sklearn.__version__},
             "timings": {"cv_seconds": cv_seconds, "final_fit_seconds": final_fit_seconds},
             "fit_seconds": perf_counter() - start}
    if rates is not None:
        model.update(training_mean=float(rates.mean()), success_labels=deepcopy(labels))
    validate_b4_model(model)
    return model


def validate_b4_model(model):
    """Reject illegal numerical/schema state in addition to the caller's hash check.

    Recomputed angles/lifts use 1e-12 absolute radian tolerance; circle norms
    also permit 1e-12 relative error across libm/platform implementations.
    Raw reference identity and discrete decisions remain exact. These checks
    do not replace the caller's exact serialized-content/hash validation.
    """
    from .features import B4_FEATURE_VERSION, B4_JOINT_TYPES, b4_feature_names

    def finite_numeric(value, shape):
        array = np.asarray(value)
        return array.shape == shape and array.dtype.kind in "if" and np.isfinite(array).all()

    def same_recomputed(left, right, *, rtol=0.):
        return np.allclose(left, right, rtol=rtol, atol=1e-12)

    try:
        json.dumps(model, allow_nan=False)
        if (model["version"] != B4_MODEL_VERSION or model["rules"] != B4_RULES or
                model["feature_version"] != B4_FEATURE_VERSION or
                model["feature_names"] != list(b4_feature_names(model["group"]))):
            raise ValueError("B4 model version, rules or feature order differs.")
        if (type(model["p"]) is not int or model["p"] not in (1, 2) or
                model["head"] not in ("angles", "success") or
                model["selected_lambda"] not in B4_LAMBDAS or
                _b4_settings(model["settings"]) != model["settings"]):
            raise ValueError("Invalid B4 model scope or hyperparameters.")
        if ((model["regime"] == "random" and model["fold"] is not None) or
                (model["regime"] == "lofo" and model["fold"] not in ("regular", "er", "ba", "sbm")) or
                model["regime"] not in ("random", "lofo")):
            raise ValueError("Invalid B4 outer scope.")
        seed = model["shuffle_seed"]
        if seed is not None and (type(seed) is not int or seed not in (20260922, 20260923, 20260924) or
                                 model["regime"] != "random" or model["group"] != "F"):
            raise ValueError("Invalid B4 null scope.")
        ids = model["training_graph_ids"]
        if not ids or ids != sorted(set(ids)):
            raise ValueError("Invalid B4 training identities.")
        n_features = len(model["feature_names"])
        n_outputs = 4 * model["p"] if model["head"] == "angles" else 1
        final = model["model"]
        if (not finite_numeric(final["coef"], (n_outputs, n_features)) or
                not finite_numeric(final["intercept"], (n_outputs,)) or final["training_graph_ids"] != ids):
            raise ValueError("Invalid B4 coefficient dimensions.")
        if any(not finite_numeric(model["timings"][key], ()) or model["timings"][key] < 0
               for key in ("cv_seconds", "final_fit_seconds")):
            raise ValueError("Invalid B4 CV/final-fit timings.")
        candidates = _b4_index(model["reference_candidates"], ids, "saved reference candidates")
        if any(not finite_numeric(row["theta"], (2 * model["p"],)) for row in candidates):
            raise ValueError("Invalid B4 saved reference dimensions.")
        reference_by_id = {row["iso_class_id"]: row for row in candidates}
        validation_ids = []
        parts = model["cv"]["folds"] + [final]
        scope = model["regime"] if model["fold"] is None else f'{model["regime"]}:{model["fold"]}'
        for part in parts:
            training = part["training_graph_ids"]
            if not training or training != sorted(set(training)) or not set(training) <= set(ids):
                raise ValueError("Invalid B4 inner training identities.")
            seen = part["seen_joint_types"]
            if len(seen) != len(B4_JOINT_TYPES) or any(type(value) is not bool for value in seen):
                raise ValueError("Invalid B4 training-fold type mask.")
            preprocessing = part["preprocessing"]
            mean, std = np.asarray(preprocessing["mean"]), np.asarray(preprocessing["std"])
            mask = preprocessing["constant_mask"]
            if (not finite_numeric(mean, (n_features,)) or not finite_numeric(std, (n_features,)) or
                    not finite_numeric(preprocessing["assortativity_median"], ()) or np.any(std < 0) or
                    len(mask) != n_features or any(type(v) is not bool for v in mask) or
                    not np.array_equal(mask, std <= 1e-12 * np.maximum(1., np.abs(mean)))):
                raise ValueError("Invalid B4 preprocessing shape or constant mask.")
            permutation = part["permutation"]
            marker = "final-fit" if part is final else part["fold"]
            expected_permutation = _b4_permutation(training, seed, scope, marker).tolist()
            if (any(type(i) is not int for i in permutation) or permutation != expected_permutation or
                    part["permuted_feature_graph_ids"] != [training[i] for i in permutation]):
                raise ValueError("Invalid B4 row permutation.")
            if model["head"] == "angles":
                anchor = part["anchor"]
                if (anchor["iso_class_id"] not in training or _angles(anchor["theta"]).size != 2 * model["p"] or
                        not same_recomputed(anchor["theta"], canonical_angles(anchor["theta"])) or
                        anchor["raw_theta"] != reference_by_id[anchor["iso_class_id"]]["theta"] or
                        not same_recomputed(anchor["theta"], canonical_angles(anchor["raw_theta"])) or
                        not finite_numeric(anchor["distance_sum"], ()) or anchor["distance_sum"] < 0 or
                        [row["iso_class_id"] for row in part["label_alignment"]] != training):
                    raise ValueError("Invalid B4 anchor or training label coverage.")
                for alignment in part["label_alignment"]:
                    original = reference_by_id[alignment["iso_class_id"]]["theta"]
                    expected = {"iso_class_id": alignment["iso_class_id"], **b4_align_label(
                        original, anchor["theta"], alignment["degree_parity"])}
                    if (set(alignment) != set(expected) or
                            any(alignment[key] != expected[key] for key in (
                                "iso_class_id", "raw_theta", "degree_parity", "mask", "sign", "tie")) or
                            type(alignment["mask"]) is not int or type(alignment["sign"]) is not int or
                            type(alignment["tie"]) is not bool or
                            not finite_numeric(alignment["lift"], (2 * model["p"],)) or
                            not finite_numeric(alignment["distance"], ()) or
                            not same_recomputed(alignment["lift"], expected["lift"]) or
                            not same_recomputed(alignment["distance"], expected["distance"])):
                        raise ValueError("B4 saved label gauge disagrees with its reference and anchor.")
            if part is not final:
                valid = part["validation_graph_ids"]
                if set(training) & set(valid) or set(training) | set(valid) != set(ids):
                    raise ValueError("B4 inner training and validation do not partition outer training.")
                validation_ids.extend(valid)
        expected_folds = model["settings"]["random_n_splits"] if model["regime"] == "random" else 3
        if len(parts) != expected_folds + 1 or sorted(validation_ids) != ids:
            raise ValueError("Invalid B4 OOF coverage.")
        if not np.array_equal(final["seen_joint_types"], np.any(
                [part["seen_joint_types"] for part in parts[:-1]], axis=0)):
            raise ValueError("B4 final type mask differs from the union of inner training masks.")
        cv = model["cv"]
        if cv["loss_graph_ids"] != ids or [row["lambda"] for row in cv["lambda_scores"]] != list(B4_LAMBDAS):
            raise ValueError("Invalid B4 CV score coverage.")
        scores = []
        for row in cv["lambda_scores"]:
            values = row["loss_by_graph"]
            if not finite_numeric(values, (len(ids),)) or min(values) < 0 or not np.isclose(
                    np.mean(values), row["mean_loss"], rtol=1e-12, atol=1e-15):
                raise ValueError("Invalid B4 CV aggregate.")
            scores.append(row["mean_loss"])
        id_index = {identity: i for i, identity in enumerate(ids)}
        validation_fold = {}
        for part in cv["folds"]:
            if [row["lambda"] for row in part["lambda_scores"]] != list(B4_LAMBDAS):
                raise ValueError("Invalid B4 inner-fold lambda score coverage.")
            indices = [id_index[identity] for identity in part["validation_graph_ids"]]
            for local, overall in zip(part["lambda_scores"], cv["lambda_scores"], strict=True):
                if not finite_numeric(local["mean_loss"], ()) or not np.isclose(local["mean_loss"],
                        np.mean(np.array(overall["loss_by_graph"])[indices]), rtol=1e-12, atol=1e-15):
                    raise ValueError("B4 inner-fold lambda score disagrees with global OOF losses.")
            validation_fold.update({identity: part for identity in part["validation_graph_ids"]})
        selected = max(value for value, score in zip(B4_LAMBDAS, scores) if score <= min(scores) + _DISTANCE_TIE)
        if selected != model["selected_lambda"]:
            raise ValueError("B4 selected lambda disagrees with CV.")
        expected_tie = sum(score <= min(scores) + _DISTANCE_TIE for score in scores) > 1
        if type(cv["lambda_tie"]) is not bool or cv["lambda_tie"] != expected_tie:
            raise ValueError("Invalid B4 lambda tie status.")
        if model["head"] == "success":
            labels = _b4_index(model["success_labels"], ids, "saved success labels")
            if (any(type(row["successes"]) is not int or type(row["trials"]) is not int or
                    row["trials"] != 50 or not 0 <= row["successes"] <= 50 for row in labels) or
                    not finite_numeric(model["training_mean"], ()) or not 0 <= model["training_mean"] <= 1 or
                    not same_recomputed(
                        model["training_mean"], np.mean([row["successes"] / 50 for row in labels]))):
                raise ValueError("Invalid B4 success baseline or observed labels.")
            rates = {row["iso_class_id"]: row["successes"] / 50 for row in labels}
        else:
            parity = {row["iso_class_id"]: row["degree_parity"] for row in final["label_alignment"]}
        predictions = cv["selected_oof_predictions"]
        if [row["iso_class_id"] for row in predictions] != ids:
            raise ValueError("B4 selected OOF predictions must cover each training graph once in ID order.")
        selected_losses = cv["lambda_scores"][B4_LAMBDAS.index(selected)]["loss_by_graph"]
        for index, prediction in enumerate(predictions):
            identity = prediction["iso_class_id"]
            if model["head"] == "angles":
                if (not finite_numeric(prediction["raw_prediction"], (4 * model["p"],)) or
                        not finite_numeric(prediction["theta0"], (2 * model["p"],)) or
                        not finite_numeric(prediction["rho"], (2 * model["p"],)) or
                        type(prediction["fallback"]) is not bool or
                        any(type(i) is not int for i in prediction["trigger_coordinates"])):
                    raise ValueError("Invalid B4 selected OOF angle diagnostic dimensions/types.")
                decoded = b4_decode_angles(prediction["raw_prediction"], validation_fold[identity]["anchor"]["theta"])
                if (any(prediction[key] != decoded[key] for key in (
                        "raw_prediction", "fallback", "trigger_coordinates")) or
                        not same_recomputed(prediction["theta0"], decoded["theta0"]) or
                        not same_recomputed(prediction["rho"], decoded["rho"], rtol=1e-12)):
                    raise ValueError("B4 selected OOF diagnostics disagree with fold-local decoding/fallback.")
                loss = b4_angle_loss(decoded["theta0"], reference_by_id[identity]["theta"], parity[identity])
            else:
                if (not finite_numeric(prediction["raw_prediction"], ()) or
                        not finite_numeric(prediction["success_probability"], ()) or
                        prediction["success_probability"] != float(np.clip(prediction["raw_prediction"], 0., 1.))):
                    raise ValueError("Invalid B4 selected OOF success prediction/clipping.")
                loss = (prediction["success_probability"] - rates[identity])**2
            if not np.isclose(loss, selected_losses[index], rtol=1e-12, atol=1e-15):
                raise ValueError("B4 selected OOF prediction disagrees with its selected-lambda loss.")
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed B4 numerical model.") from error
    return model


def predict_b4_model(model, feature_rows):
    """Predict from frozen numerical JSON and topology features only; never refit."""
    from .features import b4_feature_matrix

    validate_b4_model(model)
    if not feature_rows:
        return []
    ids = [row["iso_class_id"] for row in feature_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("B4 predictions require unique graph identities.")
    fitted = model["model"]
    matrix = _b4_transform(b4_feature_matrix(feature_rows, model["group"]), fitted["preprocessing"],
                           model["feature_names"])
    raw = matrix @ np.array(fitted["coef"]).T + fitted["intercept"]
    unseen = ~np.array(fitted["seen_joint_types"], dtype=bool)
    result = []
    for feature, prediction in zip(feature_rows, raw):
        if not np.isfinite(prediction).all():
            raise ValueError("B4 prediction is not finite.")
        if model["head"] == "angles":
            row = b4_decode_angles(prediction, fitted["anchor"]["theta"])
        else:
            row = {"success_probability": float(np.clip(prediction[0], 0., 1.)),
                   "raw_prediction": float(prediction[0])}
        counts = np.array(feature["joint_counts"])
        row.update(iso_class_id=feature["iso_class_id"],
                   unseen_type_count=int(np.count_nonzero(counts[unseen])),
                   unseen_type_fraction=float(counts[unseen].sum() / counts.sum()))
        result.append(row)
    return result
