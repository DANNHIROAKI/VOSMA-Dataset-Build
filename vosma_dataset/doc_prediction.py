"""Build DocLayNet ground truth versus a frozen, page-complete prediction run."""

import collections
import gzip
import json
import pathlib
import re
from decimal import Decimal

from .common import Dataset, load_json, sha256


TOP_K = 100
SCORE_THRESHOLD = Decimal("0.7")
INPUT_SHAPE = [1, 3, 800, 800]
EXCLUDED_FIELDS = (
    "threshold_excluded", "background_excluded", "nonfinite_score_excluded"
)


def _integer(value, description, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {description}: {value!r}")
    return value


def _physical(image):
    return (
        image["doc_category"], image["collection"],
        image["doc_name"], image["page_no"],
    )


def _decimal(value):
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _subtract_exact(left, right):
    """Subtract finite decimals without the process-wide Decimal precision."""
    a, b = left.as_tuple(), right.as_tuple()
    exponent = min(a.exponent, b.exponent)
    coefficient_a = int("".join(map(str, a.digits))) * (-1 if a.sign else 1)
    coefficient_b = int("".join(map(str, b.digits))) * (-1 if b.sign else 1)
    coefficient = (
        coefficient_a * 10 ** (a.exponent - exponent)
        - coefficient_b * 10 ** (b.exponent - exponent)
    )
    return Decimal((int(coefficient < 0), tuple(map(int, str(abs(coefficient)))), exponent))


def _xywh(prediction):
    xyxy = [_decimal(prediction[key]) for key in ("x0", "y0", "x1", "y1")]
    # Nonfinite geometry is still an exported prediction. Let Dataset.add
    # record its geometry exclusion instead of inventing a missing output.
    if not all(value.is_finite() for value in xyxy):
        return xyxy, [xyxy[0], xyxy[1], Decimal("NaN"), Decimal("NaN")]
    return xyxy, [
        xyxy[0], xyxy[1],
        _subtract_exact(xyxy[2], xyxy[0]),
        _subtract_exact(xyxy[3], xyxy[1]),
    ]


def _read_pages(path):
    manifest = load_json(path)
    if manifest.get("schema_version") != "doclaynet-pages-v1":
        raise ValueError("Unsupported DocLayNet page manifest")
    splits = manifest.get("splits")
    if (not isinstance(splits, list) or not splits
            or len(splits) != len(set(splits))
            or any(split not in ("train", "val", "test") for split in splits)):
        raise ValueError("Invalid or duplicate selected splits")
    rows = manifest.get("pages")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The page manifest must select at least one page")
    pages, identities, physicals = {}, set(), set()
    for page in rows:
        split = page["split"]
        image_id = _integer(page["source_image_id"], "source image ID")
        page_id = page["page_id"]
        if split not in splits or page_id != f"{split}/{image_id}":
            raise ValueError(f"Inconsistent page identity: {page_id!r}")
        _integer(page["width"], "page width", 1)
        _integer(page["height"], "page height", 1)
        if not isinstance(page["file_name"], str) or not page["file_name"]:
            raise ValueError(f"Missing PNG filename: {page_id}")
        physical = page["physical_key"]
        if (not isinstance(physical, list) or len(physical) != 4
                or not all(isinstance(x, str) for x in physical[:3])
                or type(physical[3]) is not int):
            raise ValueError(f"Invalid physical page identity: {page_id}")
        physical = tuple(physical)
        identity = (split, image_id)
        if page_id in pages or identity in identities or physical in physicals:
            raise ValueError(f"Duplicate selected page or physical identity: {page_id}")
        pages[page_id] = page
        identities.add(identity)
        physicals.add(physical)
    return manifest, pages


def _read_predictions(path, pages, protocol_sha256):
    results = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            result = json.loads(line, parse_float=Decimal, parse_constant=Decimal)
            page_id = result["page_id"]
            if page_id not in pages:
                raise ValueError(f"Prediction for an unselected page: {page_id}")
            if page_id in results:
                raise ValueError(f"Duplicate prediction page: {page_id}")
            if result.get("protocol_sha256") != protocol_sha256:
                raise ValueError(f"Prediction protocol hash mismatch: {page_id}")
            if result.get("status") != "success":
                raise ValueError(f"Inference did not succeed for {page_id}: {result.get('status')}")
            page = pages[page_id]
            for axis in ("width", "height"):
                if _integer(result[axis], f"prediction {axis}", 1) != page[axis]:
                    raise ValueError(f"Prediction dimensions disagree for {page_id}")
            digest = result.get("image_sha256", "")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                raise ValueError(f"Missing or invalid inference image hash: {page_id}")
            if "image_sha256" in page and digest.lower() != page["image_sha256"].lower():
                raise ValueError(f"Inference image hash mismatch: {page_id}")
            if result.get("model_input_shape") != INPUT_SHAPE:
                raise ValueError(f"Unexpected fixed model input shape: {page_id}")
            if _integer(result["top_k_candidates"], "top-k candidates") != TOP_K:
                raise ValueError(f"Unexpected fixed top-k: {page_id}")
            excluded = sum(_integer(result[key], key) for key in EXCLUDED_FIELDS)
            predictions = result.get("predictions")
            if not isinstance(predictions, list) or excluded + len(predictions) != TOP_K:
                raise ValueError(f"Inference candidate accounting mismatch: {page_id}")
            ranks = set()
            for prediction in predictions:
                rank = _integer(prediction["prediction_index"], "prediction rank")
                if rank >= TOP_K or rank in ranks:
                    raise ValueError(f"Invalid or duplicate prediction rank: {page_id}/{rank}")
                ranks.add(rank)
                if _integer(prediction["query_index"], "query index") >= 200:
                    raise ValueError(f"Query index exceeds fixed model range: {page_id}/{rank}")
                label = _integer(prediction["label"], "model label")
                if label not in range(1, 12):
                    raise ValueError(f"Exported model label violates the fixed protocol: {label}")
                score = _decimal(prediction["score"])
                if not score.is_finite() or score <= SCORE_THRESHOLD or score > 1:
                    raise ValueError(f"Exported score violates the fixed protocol: {page_id}/{rank}")
                # Parse now, before creating any output; geometry rejection
                # itself remains in Dataset.add and is counted as R2 input.
                _xywh(prediction)
            results[page_id] = (line_number, result)
    missing = set(pages) - set(results)
    if missing:
        raise ValueError(f"Missing inference pages: {sorted(missing)[:10]}")
    return results


def _check_protocol(protocol, page_manifest, pages_sha256):
    if not isinstance(protocol, dict):
        raise ValueError("The frozen model protocol must be a JSON object")
    if protocol.get("schema_version") != "doclaynet-predictions-v1":
        raise ValueError("Unsupported frozen prediction protocol schema")
    if protocol.get("page_manifest_sha256") != pages_sha256:
        raise ValueError("Protocol page manifest hash mismatch")
    if protocol.get("source_page_splits") != page_manifest["splits"]:
        raise ValueError("Protocol selected splits disagree with the page manifest")
    if protocol.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("Prediction protocol must exclude ground truth use")
    post = protocol.get("postprocessing")
    if not isinstance(post, dict):
        raise ValueError("Missing fixed prediction postprocessing protocol")
    if (_integer(post.get("top_k"), "protocol top-k") != TOP_K
            or _decimal(post.get("confidence_threshold")) != SCORE_THRESHOLD
            or post.get("threshold_operator") != ">"
            or post.get("excluded_label_ids") != [0]
            or any(post.get(key) is not False for key in ("nms", "clipping", "deduplication"))):
        raise ValueError("Prediction postprocessing differs from the fixed protocol")


def build(raw, out, pages_path, predictions_path, protocol_path):
    raw = pathlib.Path(raw)
    pages_path, predictions_path, protocol_path = map(
        pathlib.Path, (pages_path, predictions_path, protocol_path)
    )
    page_manifest, pages = _read_pages(pages_path)
    # Preserve the protocol's JSON types and values without converting its
    # numeric settings to Decimal strings in Dataset.finish.
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pages_sha256, protocol_sha256 = sha256(pages_path), sha256(protocol_path)
    _check_protocol(protocol, page_manifest, pages_sha256)
    predictions = _read_predictions(predictions_path, pages, protocol_sha256)

    selected = {(page["split"], page["source_image_id"]): page_id
                for page_id, page in pages.items()}
    ground_truth, source_provenance, categories = [], [], None
    source_counts = collections.Counter()
    for split in page_manifest["splits"]:
        path = raw / "doclaynet" / "COCO" / (split + ".json")
        source = path.relative_to(raw).as_posix()
        coco = load_json(path)
        images = {image["id"]: image for image in coco["images"]}
        if len(images) != len(coco["images"]):
            raise ValueError(f"Duplicate official image ID in {split}")
        names = {category["id"]: category["name"] for category in coco["categories"]}
        if (len(names) != len(coco["categories"]) or set(names) != set(range(1, 12))
                or any(not isinstance(name, str) or not name for name in names.values())):
            raise ValueError(f"Unexpected original DocLayNet category map in {split}")
        if categories is not None and categories != names:
            raise ValueError("DocLayNet category maps differ between selected splits")
        categories = names
        for (selected_split, image_id), page_id in selected.items():
            if selected_split != split:
                continue
            if image_id not in images:
                raise ValueError(f"Selected official image is missing: {page_id}")
            image, page = images[image_id], pages[page_id]
            if (image["file_name"] != page["file_name"]
                    or image["width"] != page["width"]
                    or image["height"] != page["height"]
                    or _physical(image) != tuple(page["physical_key"])):
                raise ValueError(f"Selected page identity differs from official GT: {page_id}")
        annotation_ids = set()
        for ordinal, annotation in enumerate(coco["annotations"]):
            source_counts["annotations.raw"] += 1
            image_id = annotation["image_id"]
            if image_id not in images:
                raise ValueError(f"Orphan official annotation: {source}:{ordinal}")
            if annotation["id"] in annotation_ids:
                raise ValueError(f"Duplicate official annotation ID: {source}:{annotation['id']}")
            annotation_ids.add(annotation["id"])
            page_id = selected.get((split, image_id))
            if page_id is None:
                source_counts["annotations.outside_selected_pages"] += 1
                continue
            if annotation.get("precedence") != 0:
                raise ValueError(f"Expected only official precedence 0 GT: {source}:{ordinal}")
            if annotation["category_id"] not in names:
                raise ValueError(f"Unknown GT class: {source}:{ordinal}")
            ground_truth.append((page_id, source, ordinal, annotation))
        source_provenance.append({
            "role": "ground_truth", "path": source, "sha256": sha256(path),
            "images": len(images), "annotations": len(coco["annotations"]),
        })

    expected_labels = {"0": "N/A", **{str(k): v for k, v in sorted(categories.items())}}
    if protocol.get("id2label") != expected_labels:
        raise ValueError("Protocol model category map differs from original DocLayNet classes")
    dataset = Dataset("doclaynet", out)
    dataset.stats.update(source_counts)
    for side in ("r1", "r2"):
        dataset.stats[f"{side}.raw"] = 0
    groups = {}
    inference_counts = collections.Counter()
    for page_id, page in pages.items():
        line_number, result = predictions[page_id]
        groups[page_id] = dataset.group(
            [page["split"], str(page["source_image_id"])],
            image_key=page_id, split=page["split"], width=page["width"], height=page["height"],
            pixel_origin=0, page_id=page_id, physical_key=page["physical_key"],
            source_image_id=page["source_image_id"], file_name=page["file_name"],
            image_sha256=result["image_sha256"], inference_status="success",
            prediction_jsonl_line=line_number,
        )
        dataset.images.append({
            "page_id": page_id, "split": page["split"],
            "source_image_id": page["source_image_id"], "file_name": page["file_name"],
            "width": page["width"], "height": page["height"],
            "image_sha256": result["image_sha256"], "inference_status": "success",
            "exported_predictions": len(result["predictions"]),
        })
        inference_counts["successful_pages"] += 1
        inference_counts["zero_output_successful_pages"] += int(not result["predictions"])
        inference_counts["top_k_candidates"] += result["top_k_candidates"]
        inference_counts["exported_predictions"] += len(result["predictions"])
        for field in EXCLUDED_FIELDS:
            inference_counts[field] += result[field]

    for page_id, source, ordinal, annotation in ground_truth:
        dataset.stats["r1.raw"] += 1
        dataset.add(
            groups[page_id], "r1", annotation.get("bbox"), source, ordinal,
            annotation_id=annotation["id"], image_id=annotation["image_id"],
            category_id=annotation["category_id"], annotation_precedence=0,
            class_name=categories[annotation["category_id"]],
        )
    prediction_ordinal = 0
    for page_id, page in pages.items():
        line_number, result = predictions[page_id]
        for prediction in result["predictions"]:
            xyxy, bbox = _xywh(prediction)
            dataset.stats["r2.raw"] += 1
            dataset.add(
                groups[page_id], "r2", bbox, predictions_path.name, prediction_ordinal,
                annotation_id=f"{page_id}:rank:{prediction['prediction_index']}",
                image_id=page["source_image_id"], category_id=prediction["label"],
                page_id=page_id, prediction_jsonl_line=line_number,
                prediction_index=prediction["prediction_index"], query_index=prediction["query_index"],
                model_label=prediction["label"], class_name=categories[prediction["label"]],
                score=str(prediction["score"]), source_xyxy=[str(value) for value in xyxy],
                image_sha256=result["image_sha256"],
            )
            prediction_ordinal += 1
    return dataset.finish({
        "source_roles": {
            "R1": "All original precedence 0 GT on selected pages; all classes",
            "R2": "Every exported fixed-model prediction; no GT matching or class filtering",
        },
        "splits": page_manifest["splits"],
        "page_counts": {"selected": len(pages), "successful_inference": len(predictions),
                        "missing": 0, "failed": 0},
        "inference_counts": dict(inference_counts),
        "model_protocol": protocol,
        "model_label_to_doclaynet_name": expected_labels,
        "sources": source_provenance,
        "prediction_provenance": {
            "pages_manifest": {"file": pages_path.name, "sha256": pages_sha256},
            "predictions": {"file": predictions_path.name, "sha256": sha256(predictions_path)},
            "protocol": {"file": protocol_path.name, "sha256": protocol_sha256},
            "image_hash_scope": "Recorded per-page inference PNG SHA256; compared to page-manifest hashes when present",
            "prediction_coordinates": "Original PNG xyxy decimal values; endpoint differences formed exactly before common quantization",
        },
        "license": "CDLA-Permissive-1.0",
    })
