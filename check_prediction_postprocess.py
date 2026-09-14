#!/usr/bin/env python3
"""Independently check one raw FP32 detector output against a prediction JSON.

Protocol: sigmoid, flatten query x label, top 100, finite score, score > 0.7,
then remove label 0. No softmax, NMS, clipping, or rounding. prediction_index
is the original top-100 rank, not a filtered row number. Only NumPy is needed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import sys

import numpy as np


class CheckError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def integer(value):
    return type(value) is int


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def sigmoid_fp32(logits):
    # Float64 intermediates avoid overflow; final scores use FP32 precision.
    # NumPy and accelerator sigmoid kernels may differ by a few FP32 ULPs.
    x = logits.astype(np.float64)
    result = np.empty_like(x)
    positive = x >= 0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        result[~positive] = exp_x / (1.0 + exp_x)
    return result.astype(np.float32)


def audit(raw_path, prediction_path, width, height):
    failures = collections.Counter()
    examples = collections.defaultdict(list)

    def check(condition, code, detail=None):
        if not condition:
            failures[code] += 1
            if len(examples[code]) < 6:
                examples[code].append(detail)
        return bool(condition)

    require(width > 0 and height > 0, "PNG width and height must be positive integers")
    with np.load(raw_path, allow_pickle=False) as tensors:
        require("logits" in tensors and "boxes" in tensors, "NPZ requires logits and boxes")
        logits, boxes = tensors["logits"], tensors["boxes"]
    require(logits.shape == (1, 200, 12), f"Unexpected logits shape: {logits.shape}")
    require(boxes.shape == (1, 200, 4), f"Unexpected boxes shape: {boxes.shape}")
    require(logits.dtype.kind == "f" and logits.dtype.itemsize == 4,
            f"Expected raw FP32 logits, got {logits.dtype}")
    require(boxes.dtype.kind == "f" and boxes.dtype.itemsize == 4,
            f"Expected raw FP32 boxes, got {boxes.dtype}")
    with prediction_path.open(encoding="utf-8") as f:
        exported = json.load(f)
    require(isinstance(exported, dict) and isinstance(exported.get("predictions"), list),
            "Prediction JSON must contain a predictions array")

    scores = sigmoid_fp32(logits[0]).reshape(-1)
    # Nonfinite sigmoid outputs are NaNs. They are excluded after top-k.
    # Without exported full candidates, NaN top-k ordering is not portable;
    # compute a diagnostic ledger but fail certification if any occur.
    ranking = np.where(np.isfinite(scores), scores.astype(np.float64), np.inf)
    order = np.argsort(-ranking, kind="stable")
    cutoff = ranking[order[99]]
    fixed = np.flatnonzero(ranking > cutoff)
    boundary = np.flatnonzero(ranking == cutoff)
    slots = 100 - len(fixed)
    fixed_set, boundary_set = set(map(int, fixed)), set(map(int, boundary))
    check(bool(np.all(np.isfinite(scores))), "nonfinite_raw_scores_prevent_portable_topk",
          int(np.sum(~np.isfinite(scores))))

    def kind(index):
        score = float(scores[index])
        if not math.isfinite(score):
            return "nonfinite_score_excluded"
        if not score > 0.7:
            return "threshold_excluded"
        return "background_excluded" if index % 12 == 0 else "retained"

    fixed_counts = collections.Counter(kind(int(i)) for i in fixed)
    boundary_counts = collections.Counter(kind(int(i)) for i in boundary)
    mandatory = {int(i) for i in fixed if kind(int(i)) == "retained"}
    allowed = mandatory | {int(i) for i in boundary if kind(int(i)) == "retained"}
    seen, seen_ranks = set(), set()
    maximum_score_error = maximum_coordinate_error = 0.0
    score_tolerance = float(4.0 * np.finfo(np.float32).eps)
    coordinate_fields = ("x0", "y0", "x1", "y1")
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    require(bool(np.all(np.isfinite(scale))), "PNG dimensions cannot be represented in FP32")
    coordinates = np.empty((200, 4), dtype=np.float32)
    coordinates[:, :2] = boxes[0, :, :2] - np.float32(0.5) * boxes[0, :, 2:]
    coordinates[:, 2:] = boxes[0, :, :2] + np.float32(0.5) * boxes[0, :, 2:]
    coordinates *= scale
    prior_rank = -1
    for row_index, row in enumerate(exported["predictions"]):
        require(isinstance(row, dict), f"Prediction {row_index} must be an object")
        q, label, rank = (row.get(k) for k in ("query_index", "label", "prediction_index"))
        if not check(integer(q) and 0 <= q < 200 and integer(label) and 0 <= label < 12,
                     "invalid_candidate_identity", {"row": row_index, "query": q, "label": label}):
            continue
        index = q * 12 + label
        check(index not in seen, "duplicate_query_label", {"query": q, "label": label})
        seen.add(index)
        check(index in allowed, "candidate_not_in_retained_top100", {"query": q, "label": label})
        if check(integer(rank) and 0 <= rank < 100, "invalid_prediction_index", rank):
            check(rank not in seen_ranks, "duplicate_prediction_index", rank)
            check(rank > prior_rank, "prediction_rows_not_in_rank_order", rank)
            seen_ranks.add(rank)
            prior_rank = rank
            # Allow arbitrary tie order and FP32 sigmoid differences in rank.
            value = float(scores[index])
            lo = int(np.sum(ranking > value + score_tolerance))
            hi = min(99, int(np.sum(ranking >= value - score_tolerance)) - 1)
            check(lo <= rank <= hi, "prediction_index_score_rank", {"rank": rank, "allowed": [lo, hi]})
        if check(finite_number(row.get("score")), "invalid_exported_score", row_index):
            error = abs(float(row["score"]) - float(scores[index]))
            maximum_score_error = max(maximum_score_error, error)
            check(error <= score_tolerance, "score_mismatch", {"row": row_index, "expected": float(scores[index]), "actual": row["score"], "absolute_error": error})
            check(row["score"] > 0.7 and label != 0, "exported_filter_violation", row_index)
        expected = coordinates[q]
        if not check(bool(np.all(np.isfinite(expected))), "nonfinite_retained_box", row_index):
            continue
        for j, field in enumerate(coordinate_fields):
            actual = row.get(field)
            if not check(finite_number(actual), "invalid_exported_coordinate", {"row": row_index, "field": field}):
                continue
            error = abs(float(actual) - float(expected[j]))
            maximum_coordinate_error = max(maximum_coordinate_error, error)
            tolerance = max(1e-6, 4.0 * abs(float(np.spacing(expected[j]))))
            check(error <= tolerance, "coordinate_mismatch", {"row": row_index, "field": field,
                  "expected": float(expected[j]), "actual": actual, "absolute_error": error, "tolerance": tolerance})

    missing = sorted(mandatory - seen)
    check(not missing, "missing_mandatory_retained_candidates", [{"query": i // 12, "label": i % 12} for i in missing[:6]])
    reported = {"retained": len(exported["predictions"])}
    for field in ("top_k_candidates", "threshold_excluded", "background_excluded", "nonfinite_score_excluded"):
        value = exported.get(field)
        check(integer(value) and value >= 0, "invalid_count", {"field": field, "value": value})
        reported[field] = value if integer(value) else -1
    check(reported["top_k_candidates"] == 100, "top_k_candidates", reported["top_k_candidates"])
    count_fields = ("retained", "threshold_excluded", "background_excluded", "nonfinite_score_excluded")
    tie_selected = {}
    for field in count_fields:
        delta = reported[field] - fixed_counts[field]
        tie_selected[field] = delta
        check(0 <= delta <= boundary_counts[field], "filter_count_mismatch",
              {"field": field, "actual": reported[field], "minimum": fixed_counts[field],
               "maximum": fixed_counts[field] + min(slots, boundary_counts[field])})
    check(sum(tie_selected.values()) == slots, "filter_partition_total", reported)
    check(tie_selected["retained"] == len(seen & boundary_set), "boundary_retained_count", tie_selected["retained"])

    finite_ordered = ranking[order]
    gap = float(finite_ordered[99] - finite_ordered[100]) if np.isfinite(finite_ordered[99:101]).all() else None
    tied_boundary = len(boundary) > slots
    near_boundary = gap is not None and gap <= 2 * score_tolerance
    near_threshold = int(np.sum(np.abs(scores.astype(np.float64) - 0.7) <= score_tolerance))
    warnings = []
    if tied_boundary:
        warnings.append("Top-100 boundary ties exist; validated a permissible tied subset and count partition, not one arbitrary tie order.")
    elif near_boundary:
        warnings.append("Top-100 boundary is within FP32 sigmoid tolerance; candidate selection may be backend-sensitive.")
    if near_threshold:
        warnings.append("Some scores are within FP32 tolerance of 0.7; filtering may be backend-sensitive.")
    return {"passed": not failures, "width": width, "height": height,
            "retained_predictions": len(exported["predictions"]), "counts": reported,
            "top100_boundary": {"score": float(cutoff) if np.isfinite(cutoff) else None,
                "strictly_above": len(fixed), "equal_score_candidates": len(boundary),
                "remaining_slots": slots, "ties_cross_boundary": tied_boundary,
                "score_gap_100_to_101": gap, "within_fp32_tolerance": near_boundary},
            "near_threshold_candidate_count": near_threshold,
            "maximum_absolute_error": {"score": maximum_score_error, "coordinate": maximum_coordinate_error},
            "tolerances": {"score_absolute": score_tolerance, "coordinate": "max(1e-6 pixel, 4 ULPs of the expected FP32 coordinate); comparisons never round values"},
            "failure_counts": dict(failures), "failure_examples": dict(examples), "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-tensors", type=pathlib.Path, required=True)
    parser.add_argument("--prediction", type=pathlib.Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve() not in {args.raw_tensors.resolve(), args.prediction.resolve()},
            "Output must not overwrite an input")
    report = {"audit_version": "prediction-postprocess-v1", "passed": False,
              "audit_script_sha256": digest(pathlib.Path(__file__).resolve()),
              "protocol": "sigmoid; flattened top100; finite score; strict score>0.7; then label!=0; cxcywh to original-PNG xyxy; no NMS, clip, rounding, or GT use"}
    try:
        report.update(audit(args.raw_tensors, args.prediction, args.width, args.height))
        report["input_sha256"] = {"raw_tensors": digest(args.raw_tensors), "prediction": digest(args.prediction)}
    except Exception as error:
        report["passed"] = False
        report["fatal_error"] = {"type": type(error).__name__, "message": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve()),
                      "failure_counts": report.get("failure_counts", {}), "fatal_error": report.get("fatal_error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
