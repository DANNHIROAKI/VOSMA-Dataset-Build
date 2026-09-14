#!/usr/bin/env python3
"""Audit complete DocLayNet PNG/prediction provenance without reading GT geometry.

Reads every selected PNG and every frozen prediction. Checks the supplied image
manifest against actual bytes, native PNG dimensions, and per-page prediction
hashes. Checkpoint manifest entries are inspected, but checkpoint files are not
inputs to this audit and are NOT independently rehashed here. Raw-tensor numeric
postprocessing verification is a separate pilot audit.
"""
from __future__ import annotations

import argparse
import collections
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import os
import pathlib
import re
import sys
import zlib


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f, parse_float=Decimal, parse_constant=Decimal)


def number(value, allow_nonfinite_string=False):
    if type(value) is int or isinstance(value, Decimal):
        return Decimal(value)
    if allow_nonfinite_string and isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation:
            raise AuditError(f"Invalid coordinate string: {value!r}") from None
        require(not parsed.is_finite(), "Finite coordinates must be JSON numbers, not strings")
        return parsed
    raise AuditError(f"Expected JSON number, got {value!r}")


def stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


class Audit:
    def __init__(self):
        self.checks = collections.defaultdict(lambda: {"checked": 0, "failed": 0})
        self.failures = collections.Counter()
        self.examples = collections.defaultdict(list)
        self.geometry = collections.Counter()
        self.checked_images = self.verified_images = self.image_bytes = 0
        self.successful_pages = self.prediction_count = 0
        self.accounting = collections.Counter()

    def check(self, condition, code, detail=None):
        self.checks[code]["checked"] += 1
        if not condition:
            self.checks[code]["failed"] += 1
            self.failures[code] += 1
            if len(self.examples[code]) < 12:
                self.examples[code].append(detail)
        return bool(condition)

    def pages(self, value):
        require(isinstance(value, dict) and value.get("schema_version") == "doclaynet-pages-v1",
                "Unsupported page manifest")
        splits = value.get("splits")
        require(isinstance(splits, list) and splits and len(splits) == len(set(splits))
                and all(s in ("train", "val", "test") for s in splits), "Invalid selected splits")
        rows = value.get("pages")
        require(isinstance(rows, list) and rows, "Page manifest must contain a nonempty pages array")
        result, names, physicals = {}, set(), set()
        for row in rows:
            require(isinstance(row, dict), "Invalid page entry")
            pid, split, iid = row.get("page_id"), row.get("split"), row.get("source_image_id")
            name, physical = row.get("file_name"), row.get("physical_key")
            require(integer(iid) and split in splits and pid == f"{split}/{iid}", "Invalid page/source identity")
            require(isinstance(name, str) and re.fullmatch(r"[0-9a-f]+\.png", name), f"Invalid PNG filename: {pid}")
            require(integer(row.get("width"), 1) and integer(row.get("height"), 1), f"Invalid image dimensions: {pid}")
            require(isinstance(physical, list) and len(physical) == 4
                    and all(isinstance(x, str) and x for x in physical[:3]) and integer(physical[3]),
                    f"Invalid physical page key: {pid}")
            physical = tuple(physical)
            self.check(pid not in result, "unique_page_identity", pid)
            self.check(name not in names, "unique_page_filename", pid)
            self.check(physical not in physicals, "unique_physical_page", pid)
            result[pid] = row
            names.add(name)
            physicals.add(physical)
        self.check(len(result) == len(rows), "page_manifest_cardinality", len(rows))
        return result

    def protocol(self, value, pages_sha, splits):
        require(isinstance(value, dict), "Invalid protocol object")
        self.check(value.get("schema_version") == "doclaynet-predictions-v1", "protocol_schema")
        self.check(value.get("page_manifest_sha256") == pages_sha, "protocol_page_manifest_sha256")
        self.check(value.get("source_page_splits") == splits, "protocol_selected_splits")
        self.check(value.get("ground_truth_used_for_prediction") is False, "protocol_declares_no_gt_use")
        post = value.get("postprocessing", {})
        expected = {"implementation": "DeformableDetrImageProcessor.post_process_object_detection",
                    "top_k": 100, "confidence_threshold": Decimal("0.7"), "threshold_operator": ">",
                    "excluded_label_ids": [0], "nms": False, "clipping": False, "deduplication": False,
                    "exclusion_order": ["nonfinite_score", "score_at_most_threshold", "N/A_background"]}
        for key, setting in expected.items():
            actual = post.get(key)
            valid = actual is setting if type(setting) is bool else actual == setting
            self.check(valid, "fixed_postprocessing_protocol", {"field": key, "actual": actual})
        infer = value.get("inference", {})
        self.check(infer.get("device") == "cpu" and infer.get("dtype") == "float32"
                   and infer.get("batch_size") == 1 and infer.get("deterministic_algorithms") is True,
                   "fixed_inference_protocol")
        self.check(integer(infer.get("threads_per_worker"), 1) and infer.get("interop_threads") == 1
                   and integer(infer.get("seed")), "recorded_inference_reproducibility")
        self.check(isinstance(value.get("preprocessing"), dict) and bool(value["preprocessing"]),
                   "recorded_preprocessing")
        labels = value.get("id2label", {})
        self.check(isinstance(labels, dict) and set(labels) == {str(i) for i in range(12)}
                   and labels.get("0") == "N/A" and all(isinstance(x, str) and x for x in labels.values())
                   and len(set(labels.values())) == 12, "recorded_complete_label_map")
        software = value.get("software", {})
        self.check(all(isinstance(software.get(k), str) and software[k]
                       for k in ("python", "torch", "torchvision", "transformers", "timm", "safetensors", "Pillow", "numpy")),
                   "recorded_software_versions")
        model = value.get("model", {})
        files = model.get("files", {}) if isinstance(model, dict) else {}
        require(isinstance(files, dict) and files, "Missing checkpoint reference manifest")
        self.check({"config.json", "preprocessor_config.json"}.issubset(files)
                   and any(name.endswith(".safetensors") for name in files), "checkpoint_manifest_required_files")
        for name, row in files.items():
            self.check(isinstance(row, dict) and is_sha(row.get("sha256")),
                       "checkpoint_manifest_hash_syntax", name)
        return {"manifest_entries_checked": len(files), "model_files_rehashed_by_this_audit": False,
                "scope": "Checkpoint manifest references in the frozen protocol were checked; model files are not inputs to this audit. The producer performs the actual model-file hash and checkpoint-load checks."}

    def image_manifest(self, value, pages, pages_sha):
        require(isinstance(value, dict), "Invalid image manifest")
        self.check(value.get("pages_sha256") == pages_sha, "image_manifest_page_sha256")
        self.check(value.get("failures") == [], "image_download_failures_empty", value.get("failures"))
        self.check(integer(value.get("selected_pages")) and value["selected_pages"] == len(pages), "image_manifest_selected_count", value.get("selected_pages"))
        self.check(isinstance(value.get("source_url"), str) and value["source_url"].startswith("https://")
                   and isinstance(value.get("archive_etag"), str) and bool(value["archive_etag"])
                   and integer(value.get("archive_size"), 1), "recorded_archive_identity")
        rows = value.get("members")
        require(isinstance(rows, list), "Missing image manifest members")
        self.check(integer(value.get("verified_images")) and value["verified_images"] == len(rows) == len(pages), "image_manifest_verified_count", len(rows))
        result, names = {}, set()
        for row in rows:
            require(isinstance(row, dict), "Invalid image manifest member")
            pid, name = row.get("page_id"), row.get("file_name")
            self.check(pid not in result, "unique_image_manifest_page", pid)
            self.check(name not in names, "unique_image_manifest_filename", name)
            require(isinstance(pid, str) and isinstance(name, str), "Invalid image manifest identity")
            result[pid] = row
            names.add(name)
            if not self.check(pid in pages, "image_manifest_selected_page", pid):
                continue
            page = pages[pid]
            self.check(name == page["file_name"] and row.get("archive_member") == "PNG/" + page["file_name"],
                       "image_manifest_member_identity", pid)
            self.check(integer(row.get("width"), 1) and integer(row.get("height"), 1)
                       and row["width"] == page["width"] and row["height"] == page["height"],
                       "image_manifest_dimensions", pid)
            self.check(is_sha(row.get("sha256")) and integer(row.get("size"), 1)
                       and isinstance(row.get("crc32"), str)
                       and re.fullmatch(r"[0-9a-fA-F]{8}", row["crc32"]) is not None,
                       "image_manifest_integrity_fields", pid)
        self.check(set(result) == set(pages), "image_manifest_exact_coverage",
                   {"missing": sorted(set(pages) - set(result))[:12], "unexpected": sorted(set(result) - set(pages))[:12]})
        return result

    def predictions(self, path, pages, protocol_sha):
        results = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    self.check(False, "prediction_nonempty_lines", lineno)
                    continue
                row = json.loads(line, parse_float=Decimal, parse_constant=Decimal)
                require(isinstance(row, dict) and isinstance(row.get("page_id"), str), f"Invalid prediction line {lineno}")
                pid = row["page_id"]
                self.check(pid not in results, "unique_prediction_page", pid)
                results[pid] = row
                if not self.check(pid in pages, "prediction_selected_page", pid):
                    continue
                page = pages[pid]
                if self.check(row.get("status") == "success", "prediction_success_status", pid):
                    self.successful_pages += 1
                self.check(row.get("protocol_sha256") == protocol_sha, "prediction_protocol_sha256", pid)
                self.check(is_sha(row.get("image_sha256")), "prediction_image_sha256_syntax", pid)
                self.check(integer(row.get("width"), 1) and integer(row.get("height"), 1)
                           and row["width"] == page["width"] and row["height"] == page["height"], "prediction_original_dimensions", pid)
                self.check(row.get("model_input_shape") == [1, 3, 800, 800], "prediction_fixed_input_shape", pid)
                preds = row.get("predictions")
                require(isinstance(preds, list), f"Missing predictions array: {pid}")
                fields = ("top_k_candidates", "nonfinite_score_excluded", "threshold_excluded", "background_excluded")
                valid_counts = all(integer(row.get(k)) and row[k] <= 100 for k in fields)
                self.check(valid_counts, "prediction_count_fields", pid)
                if valid_counts:
                    self.check(row["top_k_candidates"] == 100 and sum(row[k] for k in fields[1:]) + len(preds) == 100,
                               "prediction_candidate_accounting", pid)
                    for key in fields:
                        self.accounting[key] += row[key]
                self.prediction_count += len(preds)
                ranks, identities = set(), set()
                previous_rank, previous_score = -1, Decimal("Infinity")
                for index, prediction in enumerate(preds):
                    loc = {"page_id": pid, "row": index}
                    require(isinstance(prediction, dict), f"Invalid prediction entry: {loc}")
                    rank, query, label = (prediction.get(k) for k in ("prediction_index", "query_index", "label"))
                    if self.check(integer(rank) and rank < 100, "prediction_original_rank_range", loc):
                        self.check(rank not in ranks and rank > previous_rank, "prediction_unique_ordered_ranks", loc)
                        ranks.add(rank)
                        previous_rank = rank
                    if self.check(integer(query) and query < 200 and integer(label, 1) and label < 12,
                                  "prediction_query_and_nonbackground_label", loc):
                        identity = (query, label)
                        self.check(identity not in identities, "prediction_unique_query_label", loc)
                        identities.add(identity)
                    score = number(prediction.get("score"))
                    if self.check(score.is_finite() and Decimal("0.7") < score <= 1, "prediction_strict_score_threshold", loc):
                        self.check(score <= previous_score, "prediction_descending_scores", loc)
                        previous_score = score
                    coords = [number(prediction.get(k), allow_nonfinite_string=True) for k in ("x0", "y0", "x1", "y1")]
                    self.geometry["exported_predictions"] += 1
                    if not all(v.is_finite() for v in coords):
                        self.geometry["nonfinite_geometry"] += 1
                        continue
                    self.geometry["finite_geometry"] += 1
                    x0, y0, x1, y1 = coords
                    if x0 >= x1 or y0 >= y1:
                        self.geometry["nonpositive_extent_geometry"] += 1
                    else:
                        self.geometry["finite_positive_extent_geometry"] += 1
                    if x0 < 0 or y0 < 0:
                        self.geometry["finite_geometry_with_negative_start"] += 1
                    if x0 < 0 or y0 < 0 or x1 > page["width"] or y1 > page["height"]:
                        self.geometry["finite_geometry_outside_png"] += 1
        self.check(set(results) == set(pages), "prediction_exact_coverage",
                   {"missing": sorted(set(pages) - set(results))[:12], "unexpected": sorted(set(results) - set(pages))[:12]})
        self.check(self.successful_pages == len(pages), "successful_page_count", self.successful_pages)
        return results

    def images(self, directory, pages, members, predictions):
        from PIL import Image
        names = {p["file_name"] for p in pages.values()}
        actual = {p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
        self.check(actual == names, "png_directory_exact_coverage",
                   {"missing": sorted(names - actual)[:12], "unexpected": sorted(actual - names)[:12]})
        partials = sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix == ".partial")
        self.check(not partials, "image_download_partials_absent", partials[:12])
        aggregate = hashlib.sha256()
        for i, (pid, page) in enumerate(pages.items(), 1):
            name = page["file_name"]
            path = directory / name
            member, prediction = members.get(pid), predictions.get(pid)
            if member is None or prediction is None:
                self.check(False, "image_has_manifest_and_prediction", pid)
                continue
            before_failures = sum(self.failures.values())
            try:
                h, crc, size = hashlib.sha256(), 0, 0
                with path.open("rb") as f:
                    before = os.fstat(f.fileno())
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(block)
                        crc = zlib.crc32(block, crc)
                        size += len(block)
                    f.seek(0)
                    with Image.open(f) as image:
                        dimensions, format_name = image.size, image.format
                        image.verify()
                    after = os.fstat(f.fileno())
                    self.check(stat_identity(before) == stat_identity(after) == stat_identity(path.stat()),
                               "image_unchanged_during_verification", pid)
                self.checked_images += 1
                self.image_bytes += size
                image_sha = h.hexdigest()
                self.check(format_name == "PNG" and dimensions == (page["width"], page["height"]), "image_native_png_dimensions", pid)
                self.check(size == member.get("size") and f"{crc:08x}" == str(member.get("crc32", "")).lower(), "image_actual_size_crc32", pid)
                self.check(image_sha == str(member.get("sha256", "")).lower(), "image_actual_sha256_matches_manifest", pid)
                self.check(image_sha == str(prediction.get("image_sha256", "")).lower(), "image_actual_sha256_matches_prediction", pid)
                if "image_sha256" in page:
                    self.check(image_sha == str(page["image_sha256"]).lower(), "image_actual_sha256_matches_page_manifest", pid)
                aggregate.update(json.dumps([pid, name, image_sha, *dimensions], separators=(",", ":")).encode("utf-8") + b"\n")
                if sum(self.failures.values()) == before_failures:
                    self.verified_images += 1
            except Exception as error:
                self.check(False, "image_read_and_verify", {"page_id": pid, "error": type(error).__name__ + ": " + str(error)})
            if i % 500 == 0:
                print(json.dumps({"phase": "images", "examined": i, "selected": len(pages), "verified": self.verified_images}), flush=True)
        self.check(self.verified_images == len(pages), "all_selected_images_verified", self.verified_images)
        return aggregate.hexdigest()

    def report(self):
        geometry = {k: self.geometry[k] for k in (
            "exported_predictions", "finite_geometry", "nonfinite_geometry", "nonpositive_extent_geometry",
            "finite_positive_extent_geometry", "finite_geometry_with_negative_start", "finite_geometry_outside_png")}
        return {"successful_pages": self.successful_pages, "verified_images": self.verified_images,
                "image_files_fully_read": self.checked_images, "image_bytes_hashed": self.image_bytes,
                "prediction_count": self.prediction_count, "candidate_accounting": dict(self.accounting),
                "geometry_counts": geometry, "checks": dict(self.checks),
                "failure_counts": dict(self.failures), "failure_examples": dict(self.examples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pages", "images-manifest", "predictions", "protocol", "images", "output"):
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    args = parser.parse_args()
    input_paths = {key: getattr(args, key) for key in ("pages", "images_manifest", "predictions", "protocol")}
    require(args.output.resolve() not in {p.resolve() for p in input_paths.values()}, "Output must not overwrite an input")
    report = {"audit_version": "doclaynet-inference-provenance-v1", "passed": False,
              "audit_script_sha256": digest(pathlib.Path(__file__).resolve()),
              "selected_pages": None,
              "scope": "Actual selected PNG bytes, native dimensions, supplied source manifest, complete predictions, fixed declared protocol, and retained-score/accounting constraints. No ground-truth geometry is read.",
              "limits": "Raw logits/boxes and checkpoint files are not read here. Native numeric postprocessing is covered by the separate pilot audit; checkpoint file integrity is checked upstream. Excluded-candidate counts are reconciled arithmetically, not regenerated from logits.",
              "geometry_policy": "Nonfinite and nonpositive predicted geometry is preserved and counted for downstream builder exclusions. Negative/out-of-canvas finite coordinates are permitted. This audit does not perform Q1000 filtering."}
    audit = Audit()
    try:
        fingerprints = {key: digest(path) for key, path in input_paths.items()}
        report["input_sha256"] = fingerprints
        report["input_paths"] = {key: str(path.resolve()) for key, path in input_paths.items()}
        report["images_directory"] = str(args.images.resolve())
        pages_doc = read_json(args.pages)
        pages = audit.pages(pages_doc)
        report["selected_pages"] = len(pages)
        protocol = read_json(args.protocol)
        report["checkpoint_verification"] = audit.protocol(protocol, fingerprints["pages"], pages_doc["splits"])
        manifest = read_json(args.images_manifest)
        members = audit.image_manifest(manifest, pages, fingerprints["pages"])
        predictions = audit.predictions(args.predictions, pages, fingerprints["protocol"])
        observed_sha = audit.images(args.images, pages, members, predictions)
        report["observed_image_set_sha256"] = observed_sha
        report["verified_image_set_sha256"] = observed_sha if audit.verified_images == len(pages) else None
        for key, path in input_paths.items():
            audit.check(digest(path) == fingerprints[key], "input_unchanged_during_audit", key)
        report["image_source_verification"] = {
            "actual_png_files_fully_read": audit.checked_images,
            "all_selected_pngs_rehashed": audit.checked_images == len(pages),
            "all_selected_pngs_verified": audit.verified_images == len(pages),
            "archive_redownloaded_by_this_audit": False,
            "manifest_source_url": manifest.get("source_url"), "manifest_archive_etag": manifest.get("archive_etag"),
            "manifest_archive_size": manifest.get("archive_size"),
            "scope": "Selected local PNGs are checked against the supplied downloaded-member manifest and prediction hash; the explicit counts report completed checks. Archive identity is recorded from that manifest; this audit does not independently contact the source server."}
        report["passed"] = not audit.failures
    except Exception as error:
        report["fatal_error"] = {"type": type(error).__name__, "message": str(error)}
    report.update(audit.report())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"passed": report["passed"], "selected_pages": report["selected_pages"],
                      "successful_pages": report["successful_pages"], "verified_images": report["verified_images"],
                      "output": str(args.output.resolve()), "failure_counts": report["failure_counts"],
                      "fatal_error": report.get("fatal_error")}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
