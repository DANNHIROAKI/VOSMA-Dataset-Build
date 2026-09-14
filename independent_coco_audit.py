#!/usr/bin/env python3
"""Independently reconcile original COCO/Sama annotations and built relations.

Only quantize() is shared with the builder. Image identities, original category
mentions, filtering ledgers, and retained counts are reconstructed from raw
manifest members. This program never modifies raw data or built data.
"""
from __future__ import annotations

import argparse
import collections
from decimal import Decimal
import hashlib
import json
import pathlib
import sys
import traceback
from urllib.parse import urlsplit


class AuditError(Exception):
    pass


def read_json(path):
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_float=Decimal, parse_constant=Decimal)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value, location, positive=False):
    require(type(value) is int, f"{location}: expected integer, got {value!r}")
    require(not positive or value > 0, f"{location}: expected positive integer")
    return value


def image_name(image, location):
    filename = image.get("file_name")
    require(isinstance(filename, str) and bool(filename), f"{location}: invalid file_name")
    name = pathlib.PurePosixPath(filename).name
    require(name not in ("", ".", ".."), f"{location}: invalid file identity")
    return name


class Auditor:
    def __init__(self, raw, data, report):
        self.raw = raw.resolve()
        self.data = data.resolve()
        if not (self.data / "dataset.json").exists():
            self.data /= "coco_sama"
        self.report = report
        self.failures = collections.Counter()
        self.examples = collections.defaultdict(list)
        self.ledger = collections.Counter()
        self.groups = {}  # (split, basename, category) -> [raw1, raw2, kept1, kept2]
        self.images = {}  # (split, basename) -> side-specific original metadata/counts
        self.manifests = {}
        self.exclusions = []
        self.duplicate_ids = collections.Counter()

    def check(self, condition, code, detail=None):
        if not condition:
            self.failures[code] += 1
            if len(self.examples[code]) < 12:
                self.examples[code].append(detail)
        return condition

    def source_members(self, source):
        root = self.raw / source
        manifest_path = root / "source-manifest.json"
        manifest = read_json(manifest_path)
        members = manifest.get("members")
        require(isinstance(members, list) and members, f"{source}: empty/missing manifest members")
        paths = []
        for member in members:
            path = (self.raw / member["path"]).resolve()
            require(path.is_relative_to(root), f"{source}: member escapes source directory")
            require(path.suffix == ".json", f"{source}: unexpected non-JSON member {path}")
            require(path.is_file(), f"Missing manifest member: {path}")
            require(path not in paths, f"Duplicate manifest member: {path}")
            require(path.stat().st_size == member["size"], f"Source size mismatch: {path}")
            require(digest(path) == member["sha256"], f"Source SHA-256 mismatch: {path}")
            paths.append(path)
        actual = {p.resolve() for p in root.rglob("*.json")
                  if p.name not in {"source-manifest.json", "archive-members.json"}}
        unexpected = sorted(str(p.relative_to(self.raw)) for p in actual - set(paths))
        self.check(not unexpected, "unexpected_raw_json_files", {"source": source, "files": unexpected})
        self.manifests[source] = {
            "manifest_sha256": digest(manifest_path), "url": manifest.get("url"),
            "etag": manifest.get("etag"), "member_count": len(paths),
            "members": [{"path": str(p.relative_to(self.raw)), "sha256": m["sha256"]}
                        for p, m in zip(paths, members)],
            "unexpected_json_files": unexpected,
        }
        return sorted(paths)

    def read_side(self, split, side, files, quantize):
        side_name = ("r1", "r2")[side]
        id_to_name = {}
        metadata = {}
        categories = None
        seen_annotations = set()
        repeated_images = 0
        image_entries = 0
        duplicate_annotations = []
        side_raw = side_retained = 0
        for path in files:
            source = path.relative_to(self.raw).as_posix()
            value = read_json(path)
            require(isinstance(value, dict), f"{source}: expected COCO object")
            local_categories = {}
            for ordinal, category in enumerate(value["categories"]):
                loc = f"{source}:categories[{ordinal}]"
                cid = integer(category.get("id"), loc + ".id")
                name = category.get("name")
                require(isinstance(name, str) and bool(name), f"{loc}: invalid category name")
                require(cid not in local_categories, f"{loc}: duplicate category ID")
                require(name not in local_categories.values(), f"{loc}: duplicate category name")
                local_categories[cid] = name
            require(categories is None or local_categories == categories,
                    f"{source}: category mapping changes between shards")
            categories = local_categories
            local_images = {}
            local_names = set()
            for ordinal, image in enumerate(value["images"]):
                loc = f"{source}:images[{ordinal}]"
                iid = integer(image.get("id"), loc + ".id")
                width = integer(image.get("width"), loc + ".width", positive=True)
                height = integer(image.get("height"), loc + ".height", positive=True)
                name = image_name(image, loc)
                require(iid not in local_images, f"{loc}: duplicate image ID within file")
                require(name not in local_names, f"{loc}: duplicate filename within file")
                require(iid not in id_to_name or id_to_name[iid] == name,
                        f"{loc}: one image ID refers to multiple filenames across shards")
                current = {"id": iid, "width": width, "height": height}
                require(name not in metadata or metadata[name] == current,
                        f"{loc}: conflicting repeated image metadata")
                if "coco_url" in image:
                    url = image["coco_url"]
                    require(isinstance(url, str), f"{loc}: invalid coco_url")
                    url_path = pathlib.PurePosixPath(urlsplit(url).path)
                    require(url_path.name == name, f"{loc}: coco_url/filename mismatch")
                    require(url_path.parent.name == split + "2017", f"{loc}: coco_url split mismatch")
                repeated_images += name in metadata
                image_entries += 1
                metadata[name] = current
                id_to_name[iid] = name
                local_images[iid] = name
                local_names.add(name)
                self.images.setdefault((split, name), {"counts": [0, 0, 0, 0]})[side_name] = current
            for ordinal, annotation in enumerate(value["annotations"]):
                loc = f"{source}:annotations[{ordinal}]"
                aid = integer(annotation.get("id"), loc + ".id")
                iid = integer(annotation.get("image_id"), loc + ".image_id")
                cid = integer(annotation.get("category_id"), loc + ".category_id")
                require(iid in local_images, f"{loc}: orphan local image reference")
                require(cid in categories, f"{loc}: unknown category ID")
                if aid in seen_annotations:
                    self.duplicate_ids[f"{split}.{side_name}"] += 1
                    if len(duplicate_annotations) < 12:
                        duplicate_annotations.append({"id": aid, "source_file": source,
                                                      "source_record_index": ordinal})
                seen_annotations.add(aid)
                name = local_images[iid]
                category = categories[cid]
                key = (split, name, category)
                counts = self.groups.setdefault(key, [0, 0, 0, 0])
                counts[side] += 1
                self.images[(split, name)]["counts"][side] += 1
                self.ledger[f"{side_name}.raw"] += 1
                side_raw += 1
                crowd = annotation.get("iscrowd")
                if type(crowd) is not int or crowd not in (0, 1):
                    self.ledger[f"{side_name}.excluded.invalid_iscrowd"] += 1
                    self.check(False, "invalid_iscrowd", {"source_file": source, "row": ordinal,
                                                          "value": str(crowd)})
                    continue
                if crowd == 1:
                    self.ledger[f"{side_name}.excluded.crowd"] += 1
                    continue
                self.ledger[f"{side_name}.geometry_candidates"] += 1
                box, reason = quantize(annotation.get("bbox"))
                if reason:
                    self.ledger[f"{side_name}.excluded.{reason}"] += 1
                    self.exclusions.append({"side": side_name, "split": split,
                        "source_file": source, "source_record_index": ordinal,
                        "annotation_id": aid, "image_id": iid, "image_key": f"{split}/{name}",
                        "category_id": cid, "category": category, "reason": reason,
                        "original_bbox": annotation.get("bbox")})
                    continue
                counts[side + 2] += 1
                self.images[(split, name)]["counts"][side + 2] += 1
                self.ledger[f"{side_name}.retained"] += 1
                side_retained += 1
                if min(box[0], box[1]) < 0:
                    self.ledger[f"{side_name}.negative_local_coordinate"] += 1
                im = metadata[name]
                if box[0] < 0 or box[1] < 0 or box[2] > im["width"] * 1000 or box[3] > im["height"] * 1000:
                    self.ledger[f"{side_name}.out_of_bounds_retained"] += 1
            del value
        self.check(not duplicate_annotations, "duplicate_annotation_ids",
                   {"split": split, "side": side_name, "examples": duplicate_annotations})
        return metadata, categories, {"files": len(files), "image_metadata_entries": image_entries,
            "unique_images": len(metadata), "unique_image_ids": len(id_to_name),
            "repeated_consistent_image_metadata_entries": repeated_images,
            "duplicate_annotation_id_occurrences": self.duplicate_ids[f"{split}.{side_name}"],
            "duplicate_annotation_id_examples": duplicate_annotations,
            "raw_annotations": side_raw, "retained_annotations": side_retained}

    def raw_audit(self, quantize):
        coco_files = self.source_members("coco2017")
        require({p.name for p in coco_files} == {"instances_train2017.json", "instances_val2017.json"}
                and len(coco_files) == 2, "COCO manifest must contain exactly the train/val instance JSONs")
        splits = {}
        for split, expected in (("train", 118287), ("val", 5000)):
            left = [p for p in coco_files if p.name == f"instances_{split}2017.json"]
            right = self.source_members("sama_train" if split == "train" else "sama_val")
            cm, cc, cs = self.read_side(split, 0, left, quantize)
            sm, sc, ss = self.read_side(split, 1, right, quantize)
            cn, sn = set(cm), set(sm)
            self.check(cn == sn and len(cn) == expected, "image_coverage",
                       {"split": split, "coco": len(cn), "sama": len(sn), "expected": expected,
                        "missing_in_sama": sorted(cn - sn)[:25], "missing_in_coco": sorted(sn - cn)[:25]})
            self.check(set(cc.values()) == set(sc.values()) and len(cc) == 80,
                       "category_name_sets", {"split": split, "coco": cc, "sama": sc})
            for name in cn & sn:
                self.check((cm[name]["width"], cm[name]["height"]) == (sm[name]["width"], sm[name]["height"]),
                           "paired_image_dimensions", {"split": split, "file_name": name})
            splits[split] = {"expected_images": expected, "common_images": len(cn & sn),
                             "r1": cs, "r2": ss, "category_mapping": {"r1": cc, "r2": sc}}
        self.report["raw_splits"] = splits

    def output_audit(self, np, pq):
        info = read_json(self.data / "dataset.json")
        self.check(info.get("quantization_scale") == 1000, "quantization_scale")
        arrays = {side: np.load(self.data / (side.upper() + ".npy"), mmap_mode="r", allow_pickle=False)
                  for side in ("r1", "r2")}
        for side, array in arrays.items():
            count = self.ledger[f"{side}.retained"]
            require(array.dtype == np.dtype("<i8") and array.shape == (count, 7),
                    f"{side}: array dtype/shape differs from original retained count ({count})")
            self.check(info.get(side + "_count") == count, "dataset_relation_count", side)
            for start in range(0, count, 100000):
                block = array[start:start + 100000]
                self.check(np.array_equal(block[:, 0], np.arange(start, start + len(block))), "record_ids", side)
                self.check(bool(np.all(block[:, 6] == 1)), "unit_weights", side)
                self.check(bool(np.all(block[:, 2:4] >= 0) and np.all(block[:, 2] < block[:, 4])
                                and np.all(block[:, 3] < block[:, 5])), "array_geometry", side)
        seen_groups = set()
        offsets = [0, 0]
        states = collections.Counter()
        image_group_counts = collections.Counter()
        previous_occupied_right = None
        number = 0
        for batch in pq.ParquetFile(self.data / "groups.parquet").iter_batches(batch_size=5000):
            for row in batch.to_pylist():
                key = tuple(json.loads(row["semantic_key"]))
                require(len(key) == 3, "Invalid group semantic key")
                self.check(key not in seen_groups, "duplicate_output_group", key)
                seen_groups.add(key)
                self.check(key in self.groups, "unexpected_output_group", key)
                if key not in self.groups:
                    continue
                raw1, raw2, n1, n2 = self.groups[key]
                split, name, category = key
                image_group_counts[(split, name)] += 1
                self.check(row["group_id"] == number, "group_id_order", key)
                number += 1
                self.check(row["image_key"] == f"{split}/{name}" and row["split"] == split
                           and row["category"] == category, "group_identity_fields", key)
                im = self.images[(split, name)].get("r1", self.images[(split, name)].get("r2"))
                self.check((row["width"], row["height"]) == (im["width"], im["height"]), "group_dimensions", key)
                state = "both_nonempty" if n1 and n2 else "r1_only" if n1 else "r2_only" if n2 else "both_empty"
                self.check(row["state"] == state, "group_state", key)
                states[state] += 1
                self.check(raw1 + raw2 > 0, "fabricated_empty_group", key)
                for index, (side, count) in enumerate((("r1", n1), ("r2", n2))):
                    self.check(row[side + "_count"] == count and row[side + "_start"] == offsets[index],
                               "group_record_counts_or_offsets", {"group": key, "side": side})
                    block = arrays[side][offsets[index]:offsets[index] + count]
                    self.check(len(block) == count and bool(np.all(block[:, 1] == row["group_id"])),
                               "array_group_membership", {"group": key, "side": side})
                    if count:
                        self.check(bool(np.all(block[:, 2] >= row["global_min_x"])
                                        and np.all(block[:, 4] <= row["global_max_x"])), "array_group_slab", key)
                    offsets[index] += count
                if n1 or n2:
                    self.check(row["global_min_x"] < row["global_max_x"], "invalid_occupied_slab", key)
                    self.check(previous_occupied_right is None or previous_occupied_right < row["global_min_x"],
                               "cross_group_slab_overlap", key)
                    previous_occupied_right = row["global_max_x"]
        missing = set(self.groups) - seen_groups
        self.check(not missing, "missing_original_mention_groups", {"count": len(missing), "examples": sorted(missing)[:25]})
        self.check(len(seen_groups) == len(self.groups) == info.get("group_count"), "group_count",
                   {"raw": len(self.groups), "output": len(seen_groups), "manifest": info.get("group_count")})
        self.check(dict(states) == info.get("group_states"), "group_state_totals", dict(states))
        pairs = sum(v[2] * v[3] for v in self.groups.values())
        self.check(info.get("within_group_cartesian_pairs") == pairs,
                   "within_group_cartesian_pairs", pairs)
        self.check(offsets == [len(arrays["r1"]), len(arrays["r2"])], "array_group_total_counts", offsets)
        seen_images = set()
        table_totals = [0, 0, 0, 0]
        empty_image_stats = collections.Counter()
        for batch in pq.ParquetFile(self.data / "images.parquet").iter_batches(batch_size=5000):
            for row in batch.to_pylist():
                key = (row["split"], row["file_name"])
                self.check(key not in seen_images, "duplicate_output_image", key)
                seen_images.add(key)
                require(key in self.images, f"Unexpected image row: {key}")
                original = self.images[key]
                require("r1" in original and "r2" in original, f"Output contains unpaired image: {key}")
                counts = original["counts"]
                self.check(row["image_key"] == "/".join(key), "image_identity_fields", key)
                for side in ("r1", "r2"):
                    self.check(row[side + "_image_id"] == str(original[side]["id"]), "image_source_ids", key)
                    self.check((row["width"], row["height"]) == (original[side]["width"], original[side]["height"]),
                               "image_table_dimensions", key)
                for index, field in enumerate(("r1_raw", "r2_raw", "r1_retained", "r2_retained")):
                    self.check(row[field] == counts[index], "image_annotation_counts", {"image": key, "field": field})
                    table_totals[index] += row[field]
                raw_empty, kept_empty = not (counts[0] or counts[1]), not (counts[2] or counts[3])
                self.check(row["no_raw_annotations_on_both_sides"] == raw_empty, "image_raw_empty_flag", key)
                self.check(row["no_retained_boxes_on_both_sides"] == kept_empty, "image_retained_empty_flag", key)
                self.check(not raw_empty or image_group_counts[key] == 0, "annotation_free_image_has_groups", key)
                if "group_count" in row:
                    self.check(row["group_count"] == image_group_counts[key], "image_group_count", key)
                empty_image_stats[f"{key[0]}.no_raw_annotations_on_both_sides"] += raw_empty
                empty_image_stats[f"{key[0]}.no_retained_boxes_on_both_sides"] += kept_empty
        self.check(seen_images == set(self.images), "image_table_coverage",
                   {"raw": len(self.images), "output": len(seen_images),
                    "missing": sorted(set(self.images) - seen_images)[:25]})
        expected_totals = [self.ledger[k] for k in ("r1.raw", "r2.raw", "r1.retained", "r2.retained")]
        self.check(table_totals == expected_totals, "image_table_global_counts", {"actual": table_totals, "expected": expected_totals})
        output_ledger = info.get("filter_statistics", {})
        relevant = set(self.ledger) | {k for k in output_ledger if k.startswith(("r1.", "r2."))}
        for key in sorted(relevant):
            self.check(self.ledger[key] == output_ledger.get(key, 0), "filter_ledger",
                       {"field": key, "raw_recomputed": self.ledger[key], "output": output_ledger.get(key, 0)})
        for side in ("r1", "r2"):
            excluded = sum(v for k, v in self.ledger.items() if k.startswith(side + ".excluded."))
            self.check(self.ledger[side + ".raw"] == self.ledger[side + ".retained"] + excluded,
                       "raw_filter_reconciliation", side)
        self.report["reconciliation"] = {"original_mention_groups": len(self.groups),
            "output_groups": len(seen_groups), "group_states": dict(states),
            "original_images": len(self.images), "output_images": len(seen_images),
            "image_empty_statistics": dict(empty_image_stats),
            "relation_rows": {s: len(a) for s, a in arrays.items()},
            "within_group_cartesian_pairs": pairs}

    def finish_report(self):
        self.report.update({"source_manifests": self.manifests,
            "recomputed_filter_statistics": dict(sorted(self.ledger.items())),
            "noncrowd_geometry_exclusions": self.exclusions,
            "noncrowd_geometry_exclusion_count": len(self.exclusions),
            "duplicate_annotation_id_occurrences": dict(self.duplicate_ids),
            "failure_counts": dict(self.failures), "failure_examples": dict(self.examples)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=pathlib.Path)
    parser.add_argument("--data-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    report = {"audit_version": "independent-coco-sama-v1", "passed": False,
              "audit_script_sha256": digest(pathlib.Path(__file__).resolve()),
              "raw_root": str(args.raw_root.resolve()), "data_root": str(args.data_root.resolve()),
              "scope": "Raw manifest members, IDs, image/category identities, original mention groups, filtering, images table, and relation arrays",
              "shared_component": "vosma_dataset.common.quantize only; grouping and reconciliation independently reconstructed",
              "duplicate_id_policy": "Annotation IDs must be unique within each side and split. Consistent image metadata may repeat across shards.",
              "decimal_json_encoding": "Decimal values, including original bbox decimals, are emitted as exact strings"}
    audit = Auditor(args.raw_root, args.data_root, report)
    try:
        import numpy as np
        import pyarrow.parquet as pq
        from vosma_dataset.common import quantize
        audit.raw_audit(quantize)
        audit.output_audit(np, pq)
        report["passed"] = not audit.failures
    except Exception as error:
        report["fatal_error"] = {"type": type(error).__name__, "message": str(error)}
        if not isinstance(error, AuditError):
            report["fatal_error"]["traceback"] = traceback.format_exc()
    finally:
        audit.finish_report()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".partial")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps({"passed": report["passed"], "report": str(args.output.resolve()),
                      "noncrowd_geometry_exclusions": len(audit.exclusions),
                      "failure_counts": dict(audit.failures), "fatal_error": report.get("fatal_error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
