#!/usr/bin/env python3
"""Independently audit every DocLayNet relation record against frozen inputs.

Example:
  python audit_doclaynet_relations.py --raw-root raw --data-root data-staging \
    --pages runtime/doclaynet/pages.json \
    --predictions runtime/doclaynet/predictions/predictions.jsonl.gz \
    --output runtime/doclaynet/relations-audit.json

Only NumPy and PyArrow are required. No dataset-builder module is imported.
Coordinates are independently reconstructed with Decimal/Fraction arithmetic;
nearest-even rounding uses signed absolute-value arithmetic. This audits source
membership, geometry, provenance and packing, and never enumerates joins.
"""

import argparse
import collections
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np


Q = 1000
MODEL_ID = "Aryn/deformable-detr-DocLayNet"
MODEL_REVISION = "d5503a90ae08dd43565de6984a5dd7924cad2400"
WEIGHTS_SHA256 = "e3861d34685d3b36e5f38370597daf98fb2f98a852cfa5c125035057ded06809"
CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer",
           "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]
COORD_KEYS = ("local_x0", "local_y0", "local_x1", "local_y1")
EXCLUDED_SCORE_FIELDS = ("threshold_excluded", "background_excluded", "nonfinite_score_excluded")


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_float=Decimal, parse_constant=Decimal)


def json_value(text):
    return json.loads(text, parse_float=Decimal, parse_constant=Decimal)


def parquet_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=10000):
        yield from batch.to_pylist()


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid {name}")
    return value


def number(value):
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def nearest_even_q1000(value):
    scaled = value * Q
    sign = -1 if scaled < 0 else 1
    magnitude = abs(scaled)
    lower = magnitude.numerator // magnitude.denominator
    remainder = magnitude - lower
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and lower % 2 == 1):
        lower += 1
    return sign * lower


def geometry(values, xyxy=False):
    """Return local endpoints, exclusion reason, and exact source xywh."""
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None, "malformed_bbox", values
    try:
        decimals = [number(value) for value in values]
    except (InvalidOperation, ValueError, TypeError):
        return None, "malformed_bbox", values
    if not all(value.is_finite() for value in decimals):
        bbox = [decimals[0], decimals[1], Decimal("NaN"), Decimal("NaN")] if xyxy else decimals
        return None, "nonfinite_geometry", bbox
    exact = list(map(Fraction, decimals))
    if xyxy:
        x0, y0, x1, y1 = exact
        bbox = (x0, y0, x1 - x0, y1 - y0)
    else:
        x0, y0, width, height = exact
        x1, y1 = x0 + width, y0 + height
        bbox = tuple(exact)
    if x1 <= x0 or y1 <= y0:
        return None, "nonpositive_extent", bbox
    endpoints = tuple(nearest_even_q1000(value) for value in (x0, y0, x1, y1))
    if endpoints[0] >= endpoints[2] or endpoints[1] >= endpoints[3]:
        return None, "quantized_degenerate", bbox
    require(all(-(2**63 - 1) <= value <= 2**63 - 1 for value in endpoints),
            "Source quantized coordinates exceed the builder's int64 contract")
    return endpoints, None, bbox


def geometry_value(value):
    """Compare exact numeric geometry, including nonfinite exclusion evidence."""
    if isinstance(value, (list, tuple)):
        return tuple(geometry_value(item) for item in value)
    if isinstance(value, Fraction):
        return value
    try:
        dec = number(value)
    except (InvalidOperation, ValueError, TypeError):
        return ("literal", type(value).__name__, str(value))
    return Fraction(dec) if dec.is_finite() else ("nonfinite", str(dec))


def semantic_key(page):
    return json.dumps([page["split"], str(page["source_image_id"])],
                      ensure_ascii=False, separators=(",", ":"))


def physical(image):
    return tuple(image[key] for key in ("doc_category", "collection", "doc_name", "page_no"))


def read_inputs(raw_root, pages_path, predictions_path):
    page_document = read_json(pages_path)
    require(page_document.get("schema_version") == "doclaynet-pages-v1", "Unexpected page manifest schema")
    splits = page_document.get("splits")
    require(isinstance(splits, list) and splits and len(set(splits)) == len(splits)
            and set(splits) <= {"train", "val", "test"}, "Invalid selected split list")
    pages, page_order, seen_physical, seen_names = {}, [], set(), set()
    require(isinstance(page_document.get("pages"), list) and page_document["pages"], "Empty page selection")
    for page in page_document["pages"]:
        pid = page["page_id"]
        iid = integer(page["source_image_id"], "source image ID")
        require(page["split"] in splits and pid == f'{page["split"]}/{iid}', f"Bad page identity: {pid}")
        integer(page["width"], "width", 1); integer(page["height"], "height", 1)
        pk = page["physical_key"]
        require(isinstance(pk, list) and len(pk) == 4 and all(isinstance(x, str) for x in pk[:3])
                and type(pk[3]) is int, f"Bad physical key: {pid}")
        require(pid not in pages and tuple(pk) not in seen_physical and page["file_name"] not in seen_names,
                f"Repeated selected page, physical page or PNG filename: {pid}")
        pages[pid] = page; page_order.append(pid); seen_physical.add(tuple(pk)); seen_names.add(page["file_name"])
    protocol_path = predictions_path.with_name("protocol.json")
    protocol = read_json(protocol_path)
    protocol_hash = digest(protocol_path)
    require(protocol.get("schema_version") == "doclaynet-predictions-v1", "Unexpected inference protocol schema")
    require(protocol.get("page_manifest_sha256") == digest(pages_path), "Protocol page manifest hash mismatch")
    require(protocol.get("source_page_splits") == splits, "Protocol/page split mismatch")
    require(protocol.get("ground_truth_used_for_prediction") is False, "Protocol declares GT use during prediction")
    labels = {"0": "N/A", **{str(i): name for i, name in enumerate(CLASSES, 1)}}
    require(protocol.get("id2label") == labels, "Protocol class mapping differs from the fixed 11 classes")
    model = protocol.get("model", {})
    require(model.get("model_id") == MODEL_ID and model.get("revision") == MODEL_REVISION,
            "Model ID/revision differs from the fixed checkpoint")
    require(model.get("files", {}).get("model.safetensors", {}).get("sha256") == WEIGHTS_SHA256,
            "Model weight digest differs from the fixed checkpoint")
    post = protocol.get("postprocessing", {})
    require(post.get("top_k") == 100 and post.get("confidence_threshold") == Decimal("0.7")
            and post.get("threshold_operator") == ">" and post.get("excluded_label_ids") == [0]
            and all(post.get(key) is False for key in ("nms", "clipping", "deduplication")),
            "Postprocessing differs from fixed top-100, score>0.7, label!=0 protocol")
    predictions, inference_counts = {}, collections.Counter()
    with gzip.open(predictions_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            result = json_value(line); pid = result["page_id"]
            require(pid in pages and pid not in predictions, f"Unknown or repeated prediction page: {pid}")
            page = pages[pid]
            require(result.get("status") == "success" and result.get("protocol_sha256") == protocol_hash,
                    f"Unsuccessful or stale prediction page: {pid}")
            require(all(result.get(key) == page[key] for key in ("width", "height")), f"Prediction size mismatch: {pid}")
            image_hash = result.get("image_sha256")
            require(isinstance(image_hash, str) and re.fullmatch(r"[0-9a-fA-F]{64}", image_hash), f"Bad image hash: {pid}")
            if "image_sha256" in page:
                require(image_hash.lower() == page["image_sha256"].lower(), f"Input image hash mismatch: {pid}")
            require(result.get("model_input_shape") == [1, 3, 800, 800] and result.get("top_k_candidates") == 100,
                    f"Unexpected model input shape/top-k: {pid}")
            values = result.get("predictions")
            require(isinstance(values, list), f"Missing exported predictions: {pid}")
            discarded = sum(integer(result.get(key), key) for key in EXCLUDED_SCORE_FIELDS)
            require(discarded + len(values) == 100, f"Top-k candidate accounting mismatch: {pid}")
            ranks = set()
            for value in values:
                rank = integer(value["prediction_index"], "prediction rank")
                query = integer(value["query_index"], "query index")
                label = integer(value["label"], "model label")
                score = number(value["score"])
                require(rank < 100 and rank not in ranks and query < 200 and 1 <= label <= 11,
                        f"Invalid/duplicate rank, query or label: {pid}/{rank}")
                require(score.is_finite() and Decimal("0.7") < score <= 1, f"Invalid exported confidence: {pid}/{rank}")
                ranks.add(rank)
            predictions[pid] = (line_number, result)
            inference_counts["successful_pages"] += 1
            inference_counts["zero_output_successful_pages"] += int(not values)
            inference_counts["exported_predictions"] += len(values)
            inference_counts["top_k_candidates"] += 100
            for key in EXCLUDED_SCORE_FIELDS:
                inference_counts[key] += result[key]
    require(set(predictions) == set(pages), "Prediction pages do not exactly cover the page selection")
    return page_document, pages, page_order, protocol_path, protocol, predictions, inference_counts


def expected_relations(raw_root, page_document, pages, page_order, predictions, predictions_path):
    records = {side: {} for side in ("r1", "r2")}
    by_page = {pid: {"r1": [], "r2": []} for pid in pages}
    exclusions, stats, sources = {}, collections.Counter(), []
    class_counts = {side: collections.Counter() for side in records}
    selected = {(p["split"], p["source_image_id"]): pid for pid, p in pages.items()}
    raw_image_counts = {}

    def append(side, pid, source, ordinal, annotation_id, category_id, bbox, attrs, xyxy=False):
        page = pages[pid]
        local, reason, source_bbox = geometry(bbox, xyxy=xyxy)
        key = (source, ordinal)
        require(key not in records[side] and (side, *key) not in exclusions, f"Repeated source record: {side}/{key}")
        stats[f"{side}.raw"] += 1; stats[f"{side}.geometry_candidates"] += 1
        if reason:
            stats[f"{side}.excluded.{reason}"] += 1
            exclusions[(side, *key)] = {"reason": reason, "group": semantic_key(page),
                                       "annotation_id": str(annotation_id), "bbox": geometry_value(source_bbox)}
            return
        item = {"page_id": pid, "source_file": source, "source_record_index": ordinal,
                "source_annotation_id": str(annotation_id), "source_image_id": str(page["source_image_id"]),
                "source_category_id": str(category_id), "local": local,
                "source_bbox": geometry_value(source_bbox), "attributes": attrs}
        records[side][key] = item; by_page[pid][side].append(item)
        stats[f"{side}.retained"] += 1; class_counts[side][str(category_id)] += 1
        if local[0] < 0 or local[1] < 0:
            stats[f"{side}.negative_local_coordinate"] += 1
        if local[0] < 0 or local[1] < 0 or local[2] > page["width"] * Q or local[3] > page["height"] * Q:
            stats[f"{side}.out_of_bounds_retained"] += 1

    for side in records:
        stats[f"{side}.raw"] = 0
    for split in page_document["splits"]:
        path = raw_root / "doclaynet" / "COCO" / f"{split}.json"
        source_name = path.relative_to(raw_root).as_posix()
        raw = read_json(path)
        images = {image["id"]: image for image in raw["images"]}
        require(len(images) == len(raw["images"]), f"Repeated official image ID in {split}")
        raw_image_counts[split] = len(images)
        categories = {entry["id"]: entry["name"] for entry in raw["categories"]}
        require(len(categories) == len(raw["categories"]) and categories == dict(enumerate(CLASSES, 1)),
                f"Unexpected original category mapping: {split}")
        for (selected_split, iid), pid in selected.items():
            if selected_split != split:
                continue
            require(iid in images, f"Selected page missing from official source: {pid}")
            image, page = images[iid], pages[pid]
            require(all(image[key] == page[key] for key in ("file_name", "width", "height"))
                    and physical(image) == tuple(page["physical_key"]), f"Original physical page identity mismatch: {pid}")
        annotation_ids = set()
        for ordinal, annotation in enumerate(raw["annotations"]):
            aid, iid = annotation["id"], annotation["image_id"]
            require(aid not in annotation_ids and iid in images, f"Repeated/orphan source annotation: {split}/{aid}")
            annotation_ids.add(aid); stats["annotations.raw"] += 1
            pid = selected.get((split, iid))
            if pid is None:
                stats["annotations.outside_selected_pages"] += 1
                continue
            category = annotation["category_id"]
            require(annotation.get("precedence") == 0 and category in categories, f"Invalid GT precedence/class: {split}/{aid}")
            append("r1", pid, source_name, ordinal, aid, category, annotation.get("bbox"),
                   {"annotation_precedence": 0, "class_name": categories[category]})
        sources.append({"role": "ground_truth", "path": source_name, "sha256": digest(path),
                        "images": len(images), "annotations": len(raw["annotations"])})
    if page_document.get("selection") == "All images in each explicitly selected official split; no annotation-dependent selection":
        require(len(pages) == sum(raw_image_counts.values()), "Page manifest claims full splits but omits official images")
    ordinal = 0
    for pid in page_order:
        line_number, result = predictions[pid]
        for value in result["predictions"]:
            label, rank = value["label"], value["prediction_index"]
            xyxy = [number(value[key]) for key in ("x0", "y0", "x1", "y1")]
            append("r2", pid, predictions_path.name, ordinal, f"{pid}:rank:{rank}", label, xyxy,
                   {"page_id": pid, "prediction_jsonl_line": line_number, "prediction_index": rank,
                    "query_index": value["query_index"], "model_label": label, "class_name": CLASSES[label - 1],
                    "score": str(value["score"]), "source_xyxy": [str(x) for x in xyxy],
                    "image_sha256": result["image_sha256"]}, xyxy=True)
            ordinal += 1
    return records, by_page, exclusions, stats, sources, raw_image_counts, class_counts


def audit_groups(dataset_root, pages, predictions, by_page):
    rows = list(parquet_rows(dataset_root / "groups.parquet"))
    ordered_pages = sorted(pages, key=lambda pid: semantic_key(pages[pid]))
    require(len(rows) == len(ordered_pages), "Group table does not preserve every selected page")
    groups, offsets, states = {}, {"r1": 0, "r2": 0}, collections.Counter()
    cursor = 1
    for gid, (row, pid) in enumerate(zip(rows, ordered_pages)):
        page = pages[pid]; line_number, result = predictions[pid]
        boxes = [item["local"] for side in ("r1", "r2") for item in by_page[pid][side]]
        if boxes:
            bounds = (min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes))
            dx, dy = cursor - bounds[0], 1 - bounds[1]
            lo, hi = cursor, cursor + bounds[2] - bounds[0]
            cursor = hi + 1
        else:
            bounds = (0, 0, 0, 0); dx = dy = 0; lo = hi = cursor
        counts = {side: len(by_page[pid][side]) for side in offsets}
        state = ("both_nonempty" if all(counts.values()) else "r1_only" if counts["r1"]
                 else "r2_only" if counts["r2"] else "both_empty")
        expected = {"group_id": gid, "semantic_key": semantic_key(page), "image_key": pid,
                    "split": page["split"], "width": page["width"], "height": page["height"], "category": "",
                    "dx": dx, "dy": dy, "local_min_x": bounds[0], "local_min_y": bounds[1],
                    "local_max_x": bounds[2], "local_max_y": bounds[3], "global_min_x": lo,
                    "global_max_x": hi, "state": state,
                    **{f"{side}_{suffix}": value for side in offsets
                       for suffix, value in (("start", offsets[side]), ("count", counts[side]))}}
        require(all(row.get(key) == value for key, value in expected.items()), f"Group identity, bounds, translation or range mismatch: {pid}")
        attributes = {"pixel_origin": 0, "page_id": pid, "physical_key": page["physical_key"],
                      "source_image_id": page["source_image_id"], "file_name": page["file_name"],
                      "image_sha256": result["image_sha256"], "inference_status": "success",
                      "prediction_jsonl_line": line_number}
        require(json_value(row["attributes"]) == attributes, f"Group page provenance mismatch: {pid}")
        groups[pid] = row; states[state] += 1
        for side in offsets:
            offsets[side] += counts[side]
    return groups, dict(states)


def audit_records(dataset_root, side, expected, groups, class_counts):
    array_path = dataset_root / f"{side.upper()}.npy"
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    require(array.dtype.str == "<i8" and array.shape == (len(expected), 7), f"{side}: wrong array dtype or cardinality")
    remaining = set(expected)
    geometry_counts = collections.Counter()
    metadata_count = 0
    for index, row in enumerate(parquet_rows(dataset_root / f"{side.upper()}_metadata.parquet")):
        require(index < len(array), f"{side}: metadata has excess rows")
        key = (row["source_file"], row["source_record_index"])
        require(key in remaining, f"{side}: unexpected/repeated source record {key}")
        item = expected[key]; group = groups[item["page_id"]]
        require(row["record_id"] == index and row["group_id"] == group["group_id"], f"{side}: record/group ID mismatch at {index}")
        require(group[f"{side}_start"] <= index < group[f"{side}_start"] + group[f"{side}_count"],
                f"{side}: record lies outside its page's contiguous range at {index}")
        for field in ("source_file", "source_record_index", "source_annotation_id", "source_image_id", "source_category_id"):
            require(row[field] == item[field], f"{side}: {field} mismatch at record {index}")
        local = tuple(row[field] for field in COORD_KEYS)
        require(local == item["local"], f"{side}: independently quantized original endpoints differ at record {index}")
        require(geometry_value(json_value(row["source_bbox"])) == item["source_bbox"], f"{side}: source bbox metadata differs at record {index}")
        require(json_value(row["attributes"]) == item["attributes"], f"{side}: class, confidence, query, rank or provenance metadata differs at record {index}")
        actual = tuple(int(value) for value in array[index])
        translated = (index, group["group_id"], local[0] + group["dx"], local[1] + group["dy"],
                      local[2] + group["dx"], local[3] + group["dy"], 1)
        require(actual == translated, f"{side}: array coordinates/weight or shared translation differs at record {index}")
        require(tuple(actual[i + 2] - (group["dx"] if i % 2 == 0 else group["dy"]) for i in range(4)) == item["local"],
                f"{side}: inverse translated source endpoints differ at record {index}")
        geometry_counts[(item["page_id"], *local)] += 1
        remaining.remove(key); metadata_count += 1
    require(not remaining and metadata_count == len(array), f"{side}: missing source records or metadata rows")
    return {"npy_rows_checked": len(array), "parquet_rows_checked": metadata_count,
            "source_membership_coverage": "complete", "original_q1000_endpoint_coverage": "complete",
            "source_metadata_coverage": "complete", "retained_class_counts": dict(sorted(class_counts.items())),
            "distinct_page_geometry_tuples": len(geometry_counts),
            "duplicate_geometry_records_preserved": sum(count - 1 for count in geometry_counts.values())}


def audit_images(dataset_root, pages, predictions):
    remaining = set(pages)
    for row in parquet_rows(dataset_root / "images.parquet"):
        pid = row["page_id"]
        require(pid in remaining, f"Unknown or repeated images.parquet page: {pid}")
        page = pages[pid]; result = predictions[pid][1]
        expected = {key: page[key] for key in ("page_id", "split", "source_image_id", "file_name", "width", "height")}
        expected.update(image_sha256=result["image_sha256"], inference_status="success", exported_predictions=len(result["predictions"]))
        require(row == expected, f"Image page identity/provenance differs: {pid}")
        remaining.remove(pid)
    require(not remaining, "Images metadata omits selected pages")


def audit_exclusions(dataset_root, expected):
    remaining = set(expected)
    counts = collections.Counter()
    with gzip.open(dataset_root / "exclusions.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            require(bool(line.strip()), "Blank exclusion record")
            row = json_value(line)
            key = (row["side"], row["source_file"], row["source_record_index"])
            require(key in remaining, f"Unexpected/repeated exclusion: {key}")
            item = expected[key]
            require(all(row.get(field) == item[field] for field in ("reason", "group", "annotation_id"))
                    and geometry_value(row.get("bbox")) == item["bbox"], f"Incorrect geometry exclusion evidence: {key}")
            counts[f"{key[0]}.{item['reason']}"] += 1; remaining.remove(key)
    require(not remaining, "Missing geometry exclusion evidence")
    return dict(sorted(counts.items()))


def audit(raw_root, data_root, pages_path, predictions_path):
    dataset_root = data_root / "doclaynet"
    require(dataset_root.is_dir(), "--data-root must contain the doclaynet dataset directory")
    page_document, pages, page_order, protocol_path, protocol, predictions, inference_counts = read_inputs(raw_root, pages_path, predictions_path)
    records, by_page, exclusions, stats, sources, raw_image_counts, class_counts = expected_relations(
        raw_root, page_document, pages, page_order, predictions, predictions_path)
    print(f"INPUTS VERIFIED: {len(pages)} pages; R1={stats['r1.raw']} and R2={stats['r2.raw']} geometry candidates", flush=True)
    groups, states = audit_groups(dataset_root, pages, predictions, by_page)
    audit_images(dataset_root, pages, predictions)
    coverage = {}
    for side in ("r1", "r2"):
        coverage[side.upper()] = audit_records(dataset_root, side, records[side], groups, class_counts[side])
        coverage[side.upper()]["geometry_candidates"] = stats[f"{side}.raw"]
        print(f"{side.upper()} VERIFIED: {len(records[side])} array and metadata records", flush=True)
    excluded_counts = audit_exclusions(dataset_root, exclusions)
    info = read_json(dataset_root / "dataset.json")
    require(info.get("schema_version") == "vosma-real-rectangles-v1" and info.get("dataset") == "doclaynet"
            and info.get("quantization_scale") == Q and info.get("record_weights") == "unit"
            and info.get("no_pair_matching") is True and info.get("rectangle_semantics") == "half-open",
            "Dataset geometry/weight/membership declarations differ")
    require(info.get("r1_count") == len(records["r1"]) and info.get("r2_count") == len(records["r2"])
            and info.get("group_count") == len(pages) and info.get("group_states") == states,
            "Dataset counts/group states disagree with independent reconstruction")
    require(collections.Counter(info.get("filter_statistics", {})) == stats, "Filter statistics differ from independent source classification")
    require(collections.Counter(info.get("inference_counts", {})) == inference_counts, "Inference count summary differs from source predictions")
    require(info.get("page_counts") == {"selected": len(pages), "successful_inference": len(pages), "missing": 0, "failed": 0},
            "Page coverage declarations differ")
    require(info.get("splits") == page_document["splits"] and info.get("sources") == sources,
            "GT split/source SHA-256 provenance differs")
    require(info.get("model_protocol") == protocol and info.get("model_label_to_doclaynet_name") == protocol["id2label"],
            "Stored model protocol/class mapping differs from the frozen input")
    provenance = info.get("prediction_provenance", {})
    for field, path in (("pages_manifest", pages_path), ("predictions", predictions_path), ("protocol", protocol_path)):
        require(provenance.get(field) == {"file": path.name, "sha256": digest(path)}, f"Prediction {field} hash provenance differs")
    cartesian_count = sum(len(value["r1"]) * len(value["r2"]) for value in by_page.values())
    require(info.get("within_group_cartesian_pairs") == cartesian_count, "Declared within-group Cartesian count differs")
    outputs = ("R1.npy", "R2.npy", "R1_metadata.parquet", "R2_metadata.parquet", "groups.parquet",
               "images.parquet", "exclusions.jsonl.gz", "dataset.json")
    return {"audit_version": "doclaynet-independent-relations-v1", "passed": True, "status": "passed",
            "audit_script_sha256": digest(Path(__file__)),
            "scope": "Every selected page and every source candidate; complete NPY/Parquet/exclusion coverage; no joins enumerated",
            "independence": "No dataset-builder imports; exact Decimal/Fraction endpoints and signed nearest-even rounding",
            "pages": {"selected": len(pages), "official_pages_in_selected_splits": raw_image_counts,
                      "official_pages_not_selected": sum(raw_image_counts.values()) - len(pages),
                      "prediction_pages_checked": len(predictions), "groups_checked": len(groups),
                      "images_metadata_rows_checked": len(pages), "group_states": states},
            "relations": coverage, "geometry_exclusions_checked": len(exclusions),
            "geometry_exclusion_counts": excluded_counts, "inference_candidate_counts": dict(inference_counts),
            "membership": "All valid GT and exported predictions retained independently of opposite-side presence or class; multiplicities verified using unique source identities",
            "input_sha256": {"pages": digest(pages_path), "predictions": digest(predictions_path),
                             "protocol": digest(protocol_path), "ground_truth": {item["path"]: item["sha256"] for item in sources}},
            "output_sha256": {name: digest(dataset_root / name) for name in outputs}}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path); temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.raw_root, args.data_root, args.pages, args.predictions)
    except Exception as error:
        report = {"audit_version": "doclaynet-independent-relations-v1", "passed": False, "status": "failed",
                  "audit_script_sha256": digest(Path(__file__)),
                  "error_type": type(error).__name__, "error": str(error)}
        write_report(args.output, report)
        print(f"AUDIT FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    write_report(args.output, report)
    print("AUDIT PASSED: complete DocLayNet source membership and coordinate coverage", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
