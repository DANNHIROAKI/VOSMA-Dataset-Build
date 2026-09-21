"""Streaming, resumable Kenya footprint builder; no pair matching or deduplication.

Input files and the national boundary must be pinned by SHA-256 in a source lock.
Completed shards are immutable and hashed. A source/code/projection change makes
old checkpoints unusable. Intermediate coordinates remain signed; exactly one
shared integer translation is applied when the final two relations are written.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import pathlib
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyproj
import shapely

DATASET = "kenya_buildings"
PROJECTION = "+proj=laea +lat_0=0 +lon_0=37.5 +datum=WGS84 +x_0=0 +y_0=0 +units=m +no_defs"
Q = 1000
EXCLUSIONS_GZIP_LEVEL = 3
I64_MAX = np.iinfo(np.int64).max
COLUMNS = ["record_id", "group_id", "x0", "y0", "x1", "y1", "weight"]
META_SCHEMA = pa.schema([
    ("record_id", pa.int64()), ("group_id", pa.int64()),
    ("source_file", pa.string()), ("source_record_index", pa.int64()),
    ("source_annotation_id", pa.string()), ("source_image_id", pa.string()),
    ("source_category_id", pa.string()), ("local_x0", pa.int64()),
    ("local_y0", pa.int64()), ("local_x1", pa.int64()), ("local_y1", pa.int64()),
    ("source_bbox", pa.string()), ("attributes", pa.string()),
])
GROUP_SCHEMA = pa.schema([
    ("group_id", pa.int64()), ("semantic_key", pa.string()), ("image_key", pa.string()),
    ("split", pa.string()), ("width", pa.int64()), ("height", pa.int64()),
    ("category", pa.string()), ("dx", pa.int64()), ("dy", pa.int64()),
    ("local_min_x", pa.int64()), ("local_min_y", pa.int64()),
    ("local_max_x", pa.int64()), ("local_max_y", pa.int64()),
    ("global_min_x", pa.int64()), ("global_max_x", pa.int64()),
    ("r1_start", pa.int64()), ("r1_count", pa.int64()),
    ("r2_start", pa.int64()), ("r2_count", pa.int64()),
    ("state", pa.string()), ("attributes", pa.string()),
])


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = pathlib.Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def log(event, **fields):
    print(dumps({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "event": event, **fields}), flush=True)


def resolved_input(raw_root, relative):
    relative = pathlib.PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Input path must be relative to raw-root: {relative}")
    path = raw_root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(raw_root.resolve()):
        raise ValueError(f"Input path escapes raw-root: {relative}")
    return path


def load_lock(raw_root, lock_path):
    lock = json.loads(lock_path.read_text(encoding="utf8"))
    if lock.get("dataset") != DATASET:
        raise ValueError(f"Source lock dataset must be {DATASET!r}")
    if lock.get("crs", PROJECTION) != PROJECTION:
        raise ValueError("Source lock changes the predeclared fixed LAEA projection")
    sources = lock.get("sources", [])
    if not sources or {source["side"] for source in sources} != {"r1", "r2"}:
        raise ValueError("The source lock must contain both r1 and r2")
    paths = [source["path"] for source in sources]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate source paths in lock would duplicate records")
    for item in [lock["boundary"], *sources]:
        path = resolved_input(raw_root, item["path"])
        digest = item.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Missing/invalid SHA-256 for {item['path']}")
        if not item.get("url"):
            raise ValueError(f"Missing provenance URL for {item['path']}")
        log("verify_source", path=item["path"], bytes=path.stat().st_size)
        if sha256(path) != digest:
            raise ValueError(f"Source checksum mismatch: {item['path']}")
        if "size_bytes" in item and path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Source size mismatch: {item['path']}")
    for source in sources:
        if source["format"] not in {"geojsonl.zip", "csv.gz", "geojsonl", "csv"}:
            raise ValueError(f"Unsupported source format: {source['format']}")
    return lock


def make_transformer():
    target = pyproj.CRS.from_proj4(PROJECTION)
    return pyproj.Transformer.from_crs("EPSG:4326", target, always_xy=True)


def load_boundary(path, transformer):
    obj = json.loads(path.read_text(encoding="utf8"))
    features = obj.get("features") if obj.get("type") == "FeatureCollection" else [obj]
    if not features:
        raise ValueError("National boundary contains no geometry")
    geometries = []
    for feature in features:
        geometry = feature.get("geometry") if feature.get("type") == "Feature" else feature
        geom = shapely.from_geojson(dumps(geometry), on_invalid="raise")
        if shapely.get_type_id(geom) not in (3, 6) or shapely.is_empty(geom) or not shapely.is_valid(geom):
            raise ValueError("National boundary must contain valid Polygon/MultiPolygon geometry")
        coordinates = shapely.get_coordinates(geom)
        if not np.isfinite(coordinates).all():
            raise ValueError("National boundary has nonfinite coordinates")
        geometries.append(geom)
    boundary = shapely.union_all(geometries)
    projected = shapely.transform(boundary, transformer.transform, interleaved=False)
    if not shapely.is_valid(projected) or not np.isfinite(shapely.get_coordinates(projected)).all():
        raise ValueError("Projected national boundary is invalid")
    shapely.prepare(projected)
    return projected


@contextmanager
def input_text(raw_root, source):
    path = resolved_input(raw_root, source["path"])
    if source["format"] == "geojsonl.zip":
        with zipfile.ZipFile(path) as archive:
            members = [info.filename for info in archive.infolist() if not info.is_dir()]
            member = source.get("member")
            if member is None:
                if len(members) != 1:
                    raise ValueError(f"Pin a ZIP member in the source lock: {source['path']}")
                member = members[0]
            if member not in members:
                raise ValueError(f"Missing ZIP member: {member}")
            with archive.open(member) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream:
                    yield stream
    elif source["format"].endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
            yield stream
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield stream


def source_rows(raw_root, source) -> Iterator[dict]:
    """Yield original records, including malformed JSON; CSV parse errors are fatal."""
    with input_text(raw_root, source) as stream:
        if source["format"].startswith("geojsonl"):
            for index, line in enumerate(stream, 1):
                yield {"index": index, "line": index, "raw": line, "format": "geojson"}
        else:
            csv.field_size_limit(64 * 1024 * 1024)
            reader = csv.DictReader(stream, strict=True)
            column = source.get("geometry_column", "geometry")
            if not reader.fieldnames or column not in reader.fieldnames:
                raise ValueError(f"Missing WKT column {column!r} in {source['path']}")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Duplicate CSV header in {source['path']}")
            try:
                for index, fields in enumerate(reader, 1):
                    yield {"index": index, "line": reader.line_num,
                           "raw": fields, "geometry_column": column, "format": "wkt"}
            except csv.Error as error:
                raise ValueError(f"Malformed CSV at {source['path']} physical line {reader.line_num}: {error}") from error


def parse_rows(rows):
    geometry_texts, properties, errors, annotation_ids = [], [], [], []
    for row in rows:
        error, attrs, annotation_id, text = None, {}, "", None
        try:
            if row["format"] == "geojson":
                def reject_nonfinite(value):
                    raise ValueError(f"Nonfinite JSON number: {value}")
                obj = json.loads(row["raw"], parse_constant=reject_nonfinite)
                if not isinstance(obj, dict):
                    raise ValueError("GeoJSON record is not an object")
                if obj.get("type") == "Feature":
                    attrs = obj.get("properties") or {}
                    annotation_id = str(obj.get("id", ""))
                    geometry = obj.get("geometry")
                else:
                    geometry = obj
                # GEOS reads a Feature's geometry directly. Retain the original
                # JSON text instead of serializing every polygon a second time.
                text = row["raw"]
            else:
                if None in row["raw"] or any(value is None for value in row["raw"].values()):
                    raise ValueError("CSV row does not match header field count")
                attrs = {key: value for key, value in row["raw"].items() if key != row["geometry_column"]}
                text = row["raw"][row["geometry_column"]]
                annotation_id = str(attrs.get("full_plus_code", ""))
        except (ValueError, TypeError, KeyError) as exception:
            error = f"malformed_record: {str(exception)[:200]}"
        geometry_texts.append(text)
        properties.append(attrs)
        errors.append(error)
        annotation_ids.append(annotation_id)
    parser = shapely.from_geojson if rows[0]["format"] == "geojson" else shapely.from_wkt
    geometries = parser(geometry_texts, on_invalid="ignore")
    return geometries, properties, errors, annotation_ids


def quantized_bounds(bounds):
    """Project first, then IEEE-754 nearest-integer ties-to-even at Q=1000."""
    rounded = np.rint(np.asarray(bounds, dtype=np.float64) * Q)
    if not np.isfinite(rounded).all() or np.any(rounded <= -float(2**63)) or np.any(rounded >= float(2**63)):
        raise OverflowError("Quantized projected coordinates exceed signed int64")
    return rounded.astype("<i8")


def finite_geometry_mask(geometries):
    coordinates, owners = shapely.get_coordinates(geometries, return_index=True)
    finite = np.ones(len(geometries), dtype=bool)
    np.logical_and.at(finite, owners, np.isfinite(coordinates).all(axis=1))
    return finite


class ShardWriter:
    def __init__(self, directory, source, start, protocol, qa_remaining):
        self.directory = directory
        self.directory.mkdir(parents=True)
        self.source = source
        self.start = start
        self.end = start - 1
        self.protocol = protocol
        self.statistics = collections.Counter()
        self.count = 0
        self.bounds = None
        self.qa_remaining = qa_remaining
        self.qa = []
        self.rectangles = (directory / "rectangles.i64").open("wb")
        self.metadata = pq.ParquetWriter(directory / "metadata.parquet", META_SCHEMA, compression="zstd")
        self.exclusions_binary = (directory / "exclusions.jsonl.gz").open("wb")
        self.exclusions = io.TextIOWrapper(
            gzip.GzipFile(filename="", mode="wb", fileobj=self.exclusions_binary,
                          mtime=0, compresslevel=EXCLUSIONS_GZIP_LEVEL),
            encoding="utf8", newline="\n")

    def reject(self, row, reason, detail=None):
        self.statistics[f"excluded.{reason}"] += 1
        record = {"side": self.source["side"], "source_file": self.source["path"],
                  "source_record_index": row["index"], "source_line_number": row["line"], "reason": reason}
        if detail:
            record["detail"] = str(detail)[:250]
        self.exclusions.write(dumps(record) + "\n")

    def process(self, rows, transformer, boundary):
        if not rows:
            return
        self.end = rows[-1]["index"]
        self.statistics["input_records"] += len(rows)
        geometries, properties, errors, annotation_ids = parse_rows(rows)
        missing = shapely.is_missing(geometries)
        empty = shapely.is_empty(geometries)
        types = shapely.get_type_id(geometries)
        valid = shapely.is_valid(geometries)
        candidates = []
        for position, row in enumerate(rows):
            if errors[position]:
                self.reject(row, "malformed_record", errors[position])
            elif missing[position]:
                self.reject(row, "malformed_geometry")
            elif types[position] not in (3, 6):
                self.reject(row, "unsupported_geometry_type")
            elif empty[position]:
                self.reject(row, "empty_geometry")
            elif not valid[position]:
                self.reject(row, "invalid_geometry", shapely.is_valid_reason(geometries[position]))
            else:
                candidates.append(position)
        if not candidates:
            return
        indices = np.asarray(candidates, dtype=np.int64)
        selected = geometries[indices]
        original_bounds = shapely.bounds(selected)
        finite_source = finite_geometry_mask(selected) & np.isfinite(original_bounds).all(axis=1)
        projected = shapely.transform(selected, transformer.transform, interleaved=False)
        bounds = shapely.bounds(projected)
        finite_projected = finite_geometry_mask(projected) & np.isfinite(bounds).all(axis=1)
        projected_valid = shapely.is_valid(projected)
        # Do not ask GEOS to compute centroids of nonfinite projection results.
        projectable = finite_source & finite_projected & projected_valid
        centroid_x = np.full(len(selected), np.nan)
        centroid_y = np.full(len(selected), np.nan)
        covered = np.zeros(len(selected), dtype=bool)
        centroids = shapely.centroid(projected[projectable])
        centroid_x[projectable] = shapely.get_x(centroids)
        centroid_y[projectable] = shapely.get_y(centroids)
        covered[projectable] = shapely.covers(boundary, centroids)
        quantized = np.zeros((len(selected), 4), dtype="<i8")
        eligible = projectable & covered & np.isfinite(centroid_x) & np.isfinite(centroid_y)
        if np.any(eligible):
            quantized[eligible] = quantized_bounds(bounds[eligible])
        kept_rows = []
        kept_rectangles = []
        for selected_index, original_index in enumerate(indices):
            row = rows[original_index]
            if not finite_source[selected_index]:
                self.reject(row, "nonfinite_source_geometry")
                continue
            if not finite_projected[selected_index]:
                self.reject(row, "nonfinite_projected_geometry")
                continue
            if not projected_valid[selected_index]:
                self.reject(row, "invalid_projected_geometry", shapely.is_valid_reason(projected[selected_index]))
                continue
            if not math.isfinite(centroid_x[selected_index]) or not math.isfinite(centroid_y[selected_index]):
                self.reject(row, "nonfinite_projected_centroid")
                continue
            self.statistics["valid_projected_geometry"] += 1
            if not covered[selected_index]:
                self.reject(row, "centroid_outside_country")
                continue
            box = quantized[selected_index]
            if box[0] >= box[2] or box[1] >= box[3]:
                self.reject(row, "quantized_degenerate")
                continue
            local = box.tolist()
            projected_box = bounds[selected_index].tolist()
            centroid = [float(centroid_x[selected_index]), float(centroid_y[selected_index])]
            attrs = {"source_line_number": row["line"], "source_properties": properties[original_index],
                     "source_bounds_degrees": original_bounds[selected_index].tolist(),
                     "projected_centroid_m": centroid,
                     "geometry_type": "Polygon" if types[original_index] == 3 else "MultiPolygon"}
            kept_rows.append({"record_id": self.count + len(kept_rows), "group_id": 0,
                              "source_file": self.source["path"], "source_record_index": row["index"],
                              "source_annotation_id": annotation_ids[original_index], "source_image_id": "",
                              "source_category_id": "", "local_x0": local[0], "local_y0": local[1],
                              "local_x1": local[2], "local_y1": local[3],
                              "source_bbox": dumps(projected_box), "attributes": dumps(attrs)})
            kept_rectangles.append(box)
            if len(self.qa) < self.qa_remaining:
                self.qa.append({"type": "Feature", "geometry": json.loads(shapely.to_geojson(geometries[original_index])),
                                "properties": {"side": self.source["side"], "source_file": self.source["path"],
                                               "source_record_index": row["index"], "local_bbox_q1000": local,
                                               "projected_bbox_m": projected_box, "projected_centroid_m": centroid}})
        if kept_rows:
            rectangles = np.asarray(kept_rectangles, dtype="<i8")
            rectangles.tofile(self.rectangles)
            self.metadata.write_table(pa.Table.from_pylist(kept_rows, schema=META_SCHEMA))
            bounds = [int(rectangles[:, 0].min()), int(rectangles[:, 1].min()),
                      int(rectangles[:, 2].max()), int(rectangles[:, 3].max())]
            if self.bounds is None:
                self.bounds = bounds
            else:
                self.bounds = [min(self.bounds[0], bounds[0]), min(self.bounds[1], bounds[1]),
                               max(self.bounds[2], bounds[2]), max(self.bounds[3], bounds[3])]
            self.count += len(kept_rows)
            self.statistics["retained"] += len(kept_rows)

    def close(self, committed_directory):
        self.rectangles.flush()
        os.fsync(self.rectangles.fileno())
        self.rectangles.close()
        self.metadata.close()
        self.exclusions.close()
        self.exclusions_binary.close()
        atomic_json(self.directory / "qa.json", self.qa)
        files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(self.directory.iterdir()) if path.is_file()}
        state = {"protocol_sha256": self.protocol, "source_path": self.source["path"],
                 "source_sha256": self.source["sha256"], "side": self.source["side"],
                 "start_record": self.start, "end_record": self.end, "retained": self.count,
                 "local_bounds_q1000": self.bounds, "statistics": dict(self.statistics),
                 "qa_count": len(self.qa), "files": files}
        atomic_json(self.directory / "checkpoint.json", state)
        self.directory.rename(committed_directory)
        return state


def verify_checkpoint(directory, source, protocol):
    state = json.loads((directory / "checkpoint.json").read_text())
    if state["protocol_sha256"] != protocol or state["source_sha256"] != source["sha256"] or state["source_path"] != source["path"]:
        raise ValueError(f"Checkpoint/source/build contract mismatch: {directory}")
    for name, expected in state["files"].items():
        path = directory / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Corrupt checkpoint file: {path}")
    if (directory / "rectangles.i64").stat().st_size != state["retained"] * 4 * 8:
        raise ValueError(f"Checkpoint rectangle count mismatch: {directory}")
    if pq.ParquetFile(directory / "metadata.parquet").metadata.num_rows != state["retained"]:
        raise ValueError(f"Checkpoint metadata count mismatch: {directory}")
    return state


def process_source(raw_root, source, source_number, work_root, protocol, transformer, boundary,
                   batch_size, shard_records, qa_samples):
    source_directory = work_root / f"source-{source_number:04d}"
    source_directory.mkdir(parents=True, exist_ok=True)
    completed = []
    cursor = 0
    qa_count = 0
    for directory in sorted(source_directory.glob("part-*")):
        state = verify_checkpoint(directory, source, protocol)
        if state["start_record"] != cursor + 1 or state["end_record"] < state["start_record"]:
            raise ValueError(f"Checkpoint record sequence is not contiguous: {directory}")
        cursor = state["end_record"]
        qa_count += state["qa_count"]
        completed.append((directory, state))
    done = source_directory / "complete.json"
    if done.exists():
        marker = json.loads(done.read_text())
        if marker["protocol_sha256"] != protocol or marker["source_sha256"] != source["sha256"] or marker["input_records"] != cursor:
            raise ValueError(f"Source completion marker mismatch: {source_directory}")
        log("source_resumed_complete", path=source["path"], input_records=cursor, shards=len(completed))
        return completed
    partial = source_directory / "incomplete"
    if partial.exists():
        shutil.rmtree(partial)
    writer = None
    batch = []
    seen = 0
    started = time.monotonic()
    log("source_start", path=source["path"], side=source["side"], resume_after_record=cursor)
    for row in source_rows(raw_root, source):
        seen = row["index"]
        if seen <= cursor:
            continue
        if writer is None:
            writer = ShardWriter(partial, source, seen, protocol, max(0, qa_samples - qa_count))
        batch.append(row)
        end_part = seen - writer.start + 1 >= shard_records
        if len(batch) >= batch_size or end_part:
            writer.process(batch, transformer, boundary)
            batch = []
        if end_part:
            committed = source_directory / f"part-{writer.start:012d}"
            state = writer.close(committed)
            completed.append((committed, state))
            qa_count += state["qa_count"]
            writer = None
            log("shard_complete", path=source["path"], input_records=seen, retained=state["retained"],
                shard_start=state["start_record"], seconds=round(time.monotonic() - started, 2),
                statistics=state["statistics"])
    if writer is not None:
        writer.process(batch, transformer, boundary)
        committed = source_directory / f"part-{writer.start:012d}"
        state = writer.close(committed)
        completed.append((committed, state))
        log("shard_complete", path=source["path"], input_records=seen, retained=state["retained"],
            shard_start=state["start_record"], seconds=round(time.monotonic() - started, 2), statistics=state["statistics"])
    if seen < cursor:
        raise ValueError(f"Input ends before resumed checkpoint: {source['path']}")
    if source.get("expected_records") is not None and seen != source["expected_records"]:
        raise ValueError(f"Record count mismatch for {source['path']}: expected {source['expected_records']}, got {seen}")
    atomic_json(done, {"protocol_sha256": protocol, "source_sha256": source["sha256"], "input_records": seen})
    log("source_complete", path=source["path"], input_records=seen,
        retained=sum(state["retained"] for _, state in completed))
    return completed


def merged_bounds(shards):
    boxes = [state["local_bounds_q1000"] for _, state in shards if state["retained"]]
    if not boxes:
        raise ValueError("No records remain after source validation and country filtering")
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def process_source_task(raw_root, source, number, work_root, protocol, boundary_path,
                        batch_size, shard_records, qa_samples):
    """One independent input file; scheduling cannot change output ordering."""
    transformer = make_transformer()
    boundary = load_boundary(boundary_path, transformer)
    return process_source(raw_root, source, number, work_root, protocol, transformer, boundary,
                          batch_size, shard_records, qa_samples)


def finalize(output_root, lock_path, lock, shards, protocol, batch_size):
    target = output_root / DATASET
    output_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        info_path = target / "dataset.json"
        if info_path.exists() and json.loads(info_path.read_text()).get("build_protocol_sha256") == protocol:
            manifest = target / "SHA256SUMS"
            if not manifest.exists():
                raise ValueError("Existing output is incomplete; move it before rebuilding")
            for line in manifest.read_text().splitlines():
                digest, name = line.split("  ", 1)
                if sha256(target / name) != digest:
                    raise ValueError(f"Existing output checksum mismatch: {name}")
            log("output_already_complete", path=str(target))
            return target
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    partial = output_root / (DATASET + ".incomplete")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    counts = {side: sum(state["retained"] for _, state in shards if state["side"] == side) for side in ("r1", "r2")}
    if not all(counts.values()):
        raise ValueError(f"Both source relations must be nonempty: {counts}")
    bounds = merged_bounds(shards)
    dx, dy = 1 - bounds[0], 1 - bounds[1]
    global_bounds = [1, 1, bounds[2] + dx, bounds[3] + dy]
    if max(map(abs, [dx, dy, *global_bounds])) > I64_MAX:
        raise OverflowError("Shared integer translation exceeds int64")
    statistics = collections.Counter()
    qa = []
    for side in ("r1", "r2"):
        array = np.lib.format.open_memmap(partial / (side.upper() + ".npy"), mode="w+", dtype="<i8", shape=(counts[side], 7))
        metadata = pq.ParquetWriter(partial / (side.upper() + "_metadata.parquet"), META_SCHEMA, compression="zstd")
        offset = 0
        for directory, state in shards:
            if state["side"] != side:
                continue
            statistics.update({f"{side}.{key}": value for key, value in state["statistics"].items()})
            qa.extend(json.loads((directory / "qa.json").read_text()))
            count = state["retained"]
            if count:
                local = np.memmap(directory / "rectangles.i64", dtype="<i8", mode="r", shape=(count, 4))
                for start in range(0, count, batch_size):
                    stop = min(start + batch_size, count)
                    output = array[offset + start:offset + stop]
                    output[:, 0] = np.arange(offset + start, offset + stop, dtype="<i8")
                    output[:, 1] = 0
                    output[:, 2:6] = local[start:stop]
                    output[:, [2, 4]] += dx
                    output[:, [3, 5]] += dy
                    output[:, 6] = 1
                del local
                for batch in pq.ParquetFile(directory / "metadata.parquet").iter_batches(batch_size=batch_size):
                    table = pa.Table.from_batches([batch], schema=META_SCHEMA)
                    table = table.set_column(0, META_SCHEMA.field(0), pc.add(table.column(0), offset))
                    metadata.write_table(table)
            offset += count
            log("merge_shard", side=side, completed_records=offset, total_records=counts[side])
        if offset != counts[side]:
            raise AssertionError("Final merge row count mismatch")
        array.flush()
        del array
        metadata.close()
    with (partial / "exclusions.jsonl.gz").open("wb") as stream:
        # gzip concatenated members are specified by RFC 1952 and stream cleanly.
        for directory, _ in shards:
            with (directory / "exclusions.jsonl.gz").open("rb") as source:
                shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
    group = {"group_id": 0, "semantic_key": dumps(["KEN", "national"]), "image_key": "KEN",
             "split": "national", "width": math.ceil((bounds[2] - bounds[0]) / Q),
             "height": math.ceil((bounds[3] - bounds[1]) / Q), "category": "building",
             "dx": dx, "dy": dy, "local_min_x": bounds[0], "local_min_y": bounds[1],
             "local_max_x": bounds[2], "local_max_y": bounds[3], "global_min_x": 1,
             "global_max_x": global_bounds[2], "r1_start": 0, "r1_count": counts["r1"],
             "r2_start": 0, "r2_count": counts["r2"], "state": "both_nonempty",
             "attributes": dumps({"coordinate_system": "fixed LAEA metres, Q=1000",
                                  "single_national_group": True, "tile_boundaries_are_not_groups": True})}
    pq.write_table(pa.Table.from_pylist([group], schema=GROUP_SCHEMA), partial / "groups.parquet", compression="zstd")
    shutil.copyfile(lock_path, partial / "SOURCE_LOCK.json")
    shutil.copyfile(resolved_input(pathlib.Path(lock["_raw_root"]), lock["boundary"]["path"]), partial / "country_boundary.geojson")
    atomic_json(partial / "qa_samples.geojson", {"type": "FeatureCollection", "features": qa})
    info = {"schema_version": "vosma-real-rectangles-v1", "dataset": DATASET,
            "quantization_scale": Q, "quantization": "Projected binary64 endpoints multiplied by 1000; nearest integer, ties to even (numpy.rint)",
            "columns": COLUMNS, "dtype": "little-endian int64", "rectangle_semantics": "half-open",
            "record_weights": "unit", "r1_count": counts["r1"], "r2_count": counts["r2"], "group_count": 1,
            "group_states": {"both_nonempty": 1}, "within_group_cartesian_pairs": counts["r1"] * counts["r2"],
            "group_packing": "One national group; one shared integer translation after quantization of all retained R1/R2 bounds",
            "projection": PROJECTION, "projection_wkt": make_transformer().target_crs.to_wkt(),
            "source_crs": "EPSG:4326", "projection_datum": "WGS84", "always_xy": True,
            "country_filter": "Projected full-geometry centroid covered by the same fixed projected national boundary; full footprints retained without clipping",
            "coordinate_offset": {"dx": dx, "dy": dy}, "local_bounds_q1000": bounds,
            "global_bounds_q1000": global_bounds, "filter_statistics": dict(sorted(statistics.items())),
            "source_lock_sha256": sha256(lock_path), "build_protocol_sha256": protocol,
            "no_algorithm_index": True, "no_pair_matching": True, "no_deduplication": True,
            "no_confidence_threshold": True, "no_geometry_repair": True,
            "record_order": "Frozen source lock order, then original record order; ids restart from zero per side",
            "dependencies": {"python": sys.version.split()[0], "numpy": np.__version__, "pyarrow": pa.__version__,
                             "shapely": shapely.__version__, "geos": shapely.geos_version_string,
                             "pyproj": pyproj.__version__, "proj": pyproj.proj_version_str}}
    atomic_json(partial / "dataset.json", info)
    manifest = "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(partial.iterdir()) if path.is_file())
    (partial / "SHA256SUMS").write_text(manifest, encoding="utf8")
    partial.rename(target)
    log("dataset_complete", path=str(target), r1_count=counts["r1"], r2_count=counts["r2"], coordinate_offset={"dx": dx, "dy": dy})
    return target


def build(raw_root, source_lock, output_root, work_root, batch_size=8192, shard_records=250000, qa_samples=4, workers=1):
    if batch_size < 1 or shard_records < 1 or qa_samples < 0 or workers < 1:
        raise ValueError("Batch/shard sizes and worker count must be positive; QA count must be nonnegative")
    if tuple(int(x) for x in shapely.__version__.split(".")[:2]) < (2, 1):
        raise RuntimeError("This builder requires shapely >= 2.1 for batched always_xy projection")
    raw_root, source_lock, output_root, work_root = map(pathlib.Path, (raw_root, source_lock, output_root, work_root))
    lock = load_lock(raw_root, source_lock)
    contract = {"source_lock_sha256": sha256(source_lock), "builder_sha256": sha256(__file__),
                "projection": PROJECTION, "quantization_scale": Q, "batch_size": batch_size,
                "shard_records": shard_records, "qa_samples_per_source": qa_samples,
                "exclusions_gzip_level": EXCLUSIONS_GZIP_LEVEL,
                "python": sys.version.split()[0], "numpy": np.__version__, "pyarrow": pa.__version__, "shapely": shapely.__version__,
                "pyproj": pyproj.__version__, "geos": shapely.geos_version_string, "proj": pyproj.proj_version_str}
    protocol = hashlib.sha256(dumps(contract).encode()).hexdigest()
    work_root.mkdir(parents=True, exist_ok=True)
    contract_path = work_root / "build-contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Work directory belongs to a different source/code/projection/dependency contract; choose a new work-root")
    atomic_json(contract_path, contract)
    boundary_path = resolved_input(raw_root, lock["boundary"]["path"])
    shards = []
    if workers == 1:
        transformer = make_transformer()
        boundary = load_boundary(boundary_path, transformer)
        for number, source in enumerate(lock["sources"]):
            shards.extend(process_source(raw_root, source, number, work_root, protocol, transformer, boundary,
                                         batch_size, shard_records, qa_samples))
    else:
        log("source_workers_start", workers=min(workers, len(lock["sources"])))
        # Geometry/Arrow runtimes may start threads during import. Spawn gives
        # each worker a fresh process instead of inheriting locked fork state.
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(workers, len(lock["sources"])),
                mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(process_source_task, raw_root, source, number, work_root, protocol,
                                       boundary_path, batch_size, shard_records, qa_samples)
                       for number, source in enumerate(lock["sources"])]
            try:
                # Intentionally collect in source-lock order, not completion order.
                for future in futures:
                    shards.extend(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    lock["_raw_root"] = str(raw_root.resolve())
    return finalize(output_root, source_lock, lock, shards, protocol, batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    parser.add_argument("--source-lock", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--work-root", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--shard-records", type=int, default=250000)
    parser.add_argument("--qa-samples-per-source", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent source-file processes; output order and geometry are unchanged")
    args = parser.parse_args()
    build(args.raw_root, args.source_lock, args.output_root, args.work_root,
          args.batch_size, args.shard_records, args.qa_samples_per_source, args.workers)


if __name__ == "__main__":
    main()
