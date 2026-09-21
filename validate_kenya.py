#!/usr/bin/env python3
"""Independent streaming validation of the Kenya cross-source rectangles.

Whole-population checks never build an index or enumerate pairs. The only
spatial indexes are for three predeclared 10 km diagnostic windows. Their
results must not be presented as estimates of the national join population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

Q = 1000
I64 = np.iinfo(np.int64)
LOCAL_COLUMNS = ["local_x0", "local_y0", "local_x1", "local_y1"]
REGIONS = (
    ("Nairobi", 36.8219, -1.2921),
    ("Kisumu", 34.7617, -0.0917),
    ("Turkana", 35.6000, 3.1000),
)
WINDOW_HALF_SIZE_M = 5000
SAMPLE_SEED = 20260922


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


class SourceOrder:
    """Strict per-file ordering proves uniqueness without an O(N) record set.

    The builder emits each source file once, in source order. Rejecting a
    second file block also catches duplicate shard concatenation. Equal
    geometry from distinct source records is deliberately not rejected.
    """

    def __init__(self, allowed_files=None):
        self.allowed_files = allowed_files
        self.current_file = None
        self.completed_files = set()
        self.last_index = -1
        self.last_line = -1
        self.records = 0

    def add(self, source_file, index, line):
        require(isinstance(source_file, str) and source_file, "Missing source_file")
        require(isinstance(index, int) and index >= 0, "Invalid source_record_index")
        require(isinstance(line, int) and line >= 1, "Invalid source_line_number")
        if source_file != self.current_file:
            require(self.allowed_files is None or source_file in self.allowed_files,
                    "Metadata source_file is absent from this side of the source lock")
            if self.current_file is not None:
                self.completed_files.add(self.current_file)
            require(source_file not in self.completed_files, "Repeated source-file block")
            self.current_file = source_file
            self.last_index = self.last_line = -1
        require(index > self.last_index, "Duplicate or unordered source_record_index")
        require(line > self.last_line, "Duplicate or unordered physical source line")
        self.last_index, self.last_line = index, line
        self.records += 1

    @property
    def file_count(self):
        return len(self.completed_files) + int(self.current_file is not None)


def check_records(records, offset, local, shift):
    """Validate one aligned .npy/metadata chunk without arithmetic wraparound."""
    require(records.ndim == 2 and records.shape[1] == 7, "Expected an N x 7 array")
    require(records.dtype == np.dtype("<i8"), "Expected little-endian int64")
    require(local.shape == (len(records), 4), "Metadata bounds shape mismatch")
    require(local.dtype == np.dtype("<i8"), "Metadata bounds must be int64")
    require(np.array_equal(records[:, 0], np.arange(offset, offset + len(records))),
            "record_id is not contiguous from zero")
    require(np.all(records[:, 1] == 0), "All national records must have group_id=0")
    require(np.all(records[:, 6] == 1), "All records must have weight=1")
    require(np.all(records[:, 2:6] >= 1), "Shifted endpoints must be >=1")
    require(np.all(records[:, 2] < records[:, 4]) and
            np.all(records[:, 3] < records[:, 5]), "Nonpositive rectangle area")
    require(np.all(local[:, 0] < local[:, 2]) and
            np.all(local[:, 1] < local[:, 3]), "Nonpositive local rectangle area")
    for col, delta in enumerate((shift[0], shift[1], shift[0], shift[1])):
        require(isinstance(delta, int) and I64.min <= delta <= I64.max,
                "Integer translation is outside int64")
        if delta >= 0:
            require(np.all(local[:, col] <= I64.max - delta), "Translation overflow")
        else:
            require(np.all(local[:, col] >= I64.min - delta), "Translation underflow")
        require(np.array_equal(local[:, col] + delta, records[:, col + 2]),
                "Coordinates do not use the common integer translation")


def check_metadata_rows(data, records, local, tracker, qa_pending=None):
    require(np.array_equal(data["record_id"], records[:, 0]), "Metadata record_id mismatch")
    require(np.array_equal(data["group_id"], records[:, 1]), "Metadata group_id mismatch")
    bounds = np.asarray([json.loads(value) for value in data["source_bbox"]], dtype=np.float64)
    require(bounds.shape == local.shape and np.all(np.isfinite(bounds)),
            "Invalid projected source bounds")
    scaled = bounds * Q
    require(np.all(scaled > I64.min) and np.all(scaled < I64.max),
            "Projected bounds overflow int64 quantization")
    require(np.array_equal(np.rint(scaled).astype("<i8"), local),
            "Local coordinates do not match Q1000 ties-to-even quantization")
    centroids = np.empty((len(records), 2), dtype=np.float64)
    confidence_count = 0
    for j, encoded in enumerate(data["attributes"]):
        attrs = json.loads(encoded)
        tracker.add(data["source_file"][j], data["source_record_index"][j],
                    attrs.get("source_line_number"))
        centroid = attrs.get("projected_centroid_m")
        require(isinstance(centroid, list) and len(centroid) == 2,
                "Missing projected centroid")
        centroids[j] = centroid
        if qa_pending is not None:
            pending_file = qa_pending.get(data["source_file"][j], {})
            qa = pending_file.pop(data["source_record_index"][j], None)
            if qa is not None:
                require(np.array_equal(qa["local_bbox_q1000"], local[j]),
                        "QA sample bounds do not match its published metadata record")
                require(np.allclose(qa["projected_bbox_m"], bounds[j], rtol=0, atol=1e-6),
                        "QA sample projected bounds differ from metadata")
                require(np.allclose(qa["projected_centroid_m"], centroids[j], rtol=0, atol=1e-6),
                        "QA sample centroid differs from metadata")
        confidence = attrs.get("source_properties", {}).get("confidence")
        if confidence not in (None, ""):
            require(math.isfinite(float(confidence)), "Nonfinite retained confidence")
            confidence_count += 1
    require(np.all(np.isfinite(centroids)), "Nonfinite projected centroid")
    # A (multi)polygon's area centroid lies inside its bounding box. It need
    # not lie within the polygon itself; concavities and holes are allowed.
    require(np.all(centroids >= bounds[:, :2] - 1e-6) and
            np.all(centroids <= bounds[:, 2:] + 1e-6),
            "Projected centroid lies outside projected bounding box")
    return centroids, confidence_count


class DistributionSample:
    """Bounded, reproducible uniform pair sample using independent priorities."""

    def __init__(self, capacity=50000, seed=SAMPLE_SEED):
        require(capacity >= 1, "Sample capacity must be positive")
        self.capacity = capacity
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.total = np.zeros(2, dtype=np.float64)
        self.minimum = np.full(2, np.inf)
        self.maximum = np.full(2, -np.inf)
        self.kept = np.empty((0, 3), dtype=np.float64)
        self.pending = []
        self.pending_count = 0

    def add(self, areas_m2, ious):
        require(len(areas_m2) == len(ious), "Distribution columns differ")
        if not len(ious):
            return
        values = np.column_stack((areas_m2, ious))
        require(np.all(np.isfinite(values)) and np.all(values > 0) and
                np.all(ious <= 1 + 1e-12), "Invalid positive-pair area or IoU")
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.pending.append(np.column_stack((self.rng.random(len(values)), values)))
        self.pending_count += len(values)
        if self.pending_count >= 8192:
            self._merge()

    def _merge(self):
        if not self.pending:
            return
        candidates = np.concatenate([self.kept, *self.pending])
        if len(candidates) > self.capacity:
            selected = np.argpartition(candidates[:, 0], self.capacity - 1)[:self.capacity]
            candidates = candidates[selected]
        self.kept = candidates
        self.pending.clear()
        self.pending_count = 0

    def finish(self):
        self._merge()
        result = {"positive_pairs": self.count, "quantile_sample_size": len(self.kept),
                  "quantile_method": "all pairs" if self.count <= self.capacity else
                  "uniform independent-priority sample", "sample_seed": self.seed}
        for col, label in enumerate(("intersection_area_m2", "iou")):
            result[label] = {} if not self.count else {
                "min": float(self.minimum[col]), "max": float(self.maximum[col]),
                "mean": float(self.total[col] / self.count),
                **dict(zip(("p05", "median", "p95"),
                           map(float, np.quantile(self.kept[:, col + 1], [.05, .5, .95])))),
            }
        return result


def region_join(a, b):
    """Count all positive rectangle pairs in a local subset; retain no pairs."""
    import shapely

    require(int(shapely.__version__.split(".")[0]) >= 2, "Shapely >=2 is required")
    sample = DistributionSample()
    hit_a = np.zeros(len(a), dtype=bool)
    hit_b = np.zeros(len(b), dtype=bool)
    if len(a) and len(b):
        boxes_b = shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3])
        tree = shapely.STRtree(boxes_b)
        area_b = (b[:, 2].astype(float) - b[:, 0]) * (b[:, 3].astype(float) - b[:, 1])
        for index, rect in enumerate(a):
            candidates = tree.query(shapely.box(*rect))
            if not len(candidates):
                continue
            selected = b[candidates]
            extent = np.minimum(rect[2:], selected[:, 2:]) - np.maximum(rect[:2], selected[:, :2])
            positive = np.all(extent > 0, axis=1)
            if not np.any(positive):
                continue
            ids = candidates[positive]
            hit_a[index] = True
            hit_b[ids] = True
            sizes = extent[positive].astype(float)
            intersections = sizes[:, 0] * sizes[:, 1]
            area_a = (int(rect[2]) - int(rect[0])) * (int(rect[3]) - int(rect[1]))
            ious = intersections / (area_a + area_b[ids] - intersections)
            sample.add(intersections / Q**2, ious)
    answer = sample.finish()
    answer.update(r1_count=len(a), r2_count=len(b), r1_no_overlap=int((~hit_a).sum()),
                  r2_no_overlap=int((~hit_b).sum()),
                  r1_no_overlap_fraction=float((~hit_a).mean()) if len(a) else None,
                  r2_no_overlap_fraction=float((~hit_b).mean()) if len(b) else None)
    return answer


def qa_geometry(path, projection, output_path):
    """Independently project original QA polygons, check boxes and draw overlays."""
    import pyproj
    from shapely.geometry import shape
    from shapely.ops import transform

    require(path.exists(), "Missing qa_samples.geojson")
    features = json.loads(path.read_text())["features"]
    require(features, "Empty polygon QA sample")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    projected = []
    for feature in features:
        props = feature["properties"]
        geom = transform(transformer.transform, shape(feature["geometry"]))
        require(not geom.is_empty and geom.is_valid and geom.area > 0, "Invalid QA polygon")
        qbox = np.rint(np.asarray(geom.bounds) * Q).astype("<i8")
        require(np.array_equal(qbox, props["local_bbox_q1000"]), "QA polygon projection/box mismatch")
        require(np.allclose(geom.bounds, props["projected_bbox_m"], rtol=0, atol=1e-6),
                "QA floating projected bounds mismatch")
        require(np.allclose([geom.centroid.x, geom.centroid.y], props["projected_centroid_m"],
                            rtol=0, atol=1e-6), "QA projected centroid mismatch")
        projected.append((geom, props))
    # Keep plotting caches with this build, not in a user's home directory.
    with tempfile.TemporaryDirectory(prefix=".qa-cache-", dir=output_path.parent) as cache:
        previous = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
        os.environ["MPLCONFIGDIR"] = str(Path(cache) / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(Path(cache) / "xdg")
        try:
            return _qa_overlay(projected, output_path, len(features))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _qa_overlay(projected, output_path, sample_count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    selected = []
    for side in ("r1", "r2"):
        selected.extend([(g, p) for g, p in projected if p["side"].lower() == side][:4])
    require(selected, "QA samples have no recognized side")
    fig, axes = plt.subplots(math.ceil(len(selected) / 4), 4, figsize=(12, 3 * math.ceil(len(selected) / 4)),
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (geom, props) in zip(axes.flat, selected):
        ax.set_visible(True)
        xmin, ymin, xmax, ymax = geom.bounds
        for poly in geom.geoms if geom.geom_type == "MultiPolygon" else [geom]:
            xs, ys = poly.exterior.xy
            ax.fill(np.asarray(xs) - xmin, np.asarray(ys) - ymin, color="#2563eb", alpha=.25)
            ax.plot(np.asarray(xs) - xmin, np.asarray(ys) - ymin, color="#2563eb", linewidth=.9)
            for ring in poly.interiors:
                xs, ys = ring.xy
                ax.fill(np.asarray(xs) - xmin, np.asarray(ys) - ymin, color="white")
        qx0, qy0, qx1, qy1 = np.asarray(props["local_bbox_q1000"], dtype=float) / Q
        ax.add_patch(Rectangle((qx0 - xmin, qy0 - ymin), qx1 - qx0, qy1 - qy0, fill=False,
                               edgecolor="#e25822", linewidth=1.2))
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(.15)
        ax.set_xlabel("Projected easting relative to box minimum (m)", fontsize=7)
        ax.set_ylabel("Northing (m)", fontsize=7)
        ax.set_title(f"{props['side'].upper()} source record {props['source_record_index']}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Building footprints (blue) and published Q1000 rectangles (orange)", fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"checked_original_polygons": sample_count, "overlay_polygons": len(selected),
            "overlay": output_path.name, "projection_recomputed": True,
            "scope": "Deterministic source samples; not a random national quality estimate"}


def validate(path, batch_size=50000, progress=True):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyproj
    import shapely
    from shapely.geometry import shape
    from shapely.ops import transform as transform_geometry

    path = Path(path)
    info = json.loads((path / "dataset.json").read_text())
    require(info["quantization_scale"] == Q, "Expected quantization_scale=1000")
    require(info["group_count"] == 1, "Expected a single continuous national group")
    projection = info["projection"]
    if isinstance(projection, dict):
        projection = projection.get("proj") or projection.get("proj_string") or projection.get("definition")
    require(isinstance(projection, str) and "+proj=laea" in projection, "Expected a shared LAEA projection")
    if not any(option in projection for option in ("+datum=", "+ellps=", "+R=")):
        require(info.get("projection_datum") == "WGS84", "Projection needs an explicit WGS84 datum")
        projection += " +datum=WGS84"
    crs = pyproj.CRS.from_user_input(projection)
    require(crs.is_projected and all(axis.unit_name == "metre" for axis in crs.axis_info),
            "Projection must use metres")
    shift = (info["coordinate_offset"]["dx"], info["coordinate_offset"]["dy"])
    lock_path = path / "SOURCE_LOCK.json"
    require(file_sha256(lock_path) == info["source_lock_sha256"], "Source lock hash mismatch")
    lock = json.loads(lock_path.read_text())
    boundary_path = path / "country_boundary.geojson"
    require(file_sha256(boundary_path) == lock["boundary"]["sha256"], "National boundary hash mismatch")
    transform = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    boundary_json = json.loads(boundary_path.read_text())
    if boundary_json.get("type") == "FeatureCollection":
        boundary = shapely.union_all([shape(f["geometry"]) for f in boundary_json["features"]])
    else:
        boundary = shape(boundary_json.get("geometry", boundary_json))
    require(boundary.is_valid and not boundary.is_empty, "Invalid source national boundary")
    projected_boundary = transform_geometry(transform.transform, boundary)
    require(projected_boundary.is_valid and not projected_boundary.is_empty, "Invalid projected national boundary")
    shapely.prepare(projected_boundary)
    qa_path = path / "qa_samples.geojson"
    qa_pending = {"r1": {}, "r2": {}}
    for feature in json.loads(qa_path.read_text())["features"]:
        props = feature["properties"]
        side = props["side"].lower()
        require(side in qa_pending, "Unexpected QA sample side")
        file_samples = qa_pending[side].setdefault(props["source_file"], {})
        index = props["source_record_index"]
        require(index not in file_samples, "Duplicate QA sample source record")
        file_samples[index] = props
    require(all(qa_pending.values()), "Original polygon QA samples must cover both sources")
    regions = []
    for name, lon, lat in REGIONS:
        x, y = transform.transform(lon, lat)
        regions.append({"name": name, "center_wgs84_lon_lat": [lon, lat],
                        "center_projected_m": [x, y],
                        "window_projected_m": [x - WINDOW_HALF_SIZE_M, y - WINDOW_HALF_SIZE_M,
                                                x + WINDOW_HALF_SIZE_M, y + WINDOW_HALF_SIZE_M],
                        "r1": [], "r2": []})
    results = {}
    global_low = np.full(2, I64.max, dtype="<i8")
    global_high = np.full(2, I64.min, dtype="<i8")
    max_coordinate = 0
    for side in ("r1", "r2"):
        array_path = path / f"{side.upper()}.npy"
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        require(array.dtype == np.dtype("<i8") and array.shape == (info[f"{side}_count"], 7),
                f"{side} NPY schema/count mismatch")
        require(len(array) > 0, f"{side} is empty")
        parquet = pq.ParquetFile(path / f"{side.upper()}_metadata.parquet")
        require(parquet.metadata.num_rows == len(array), f"{side} metadata count mismatch")
        for column in ["record_id", "group_id", "source_record_index", *LOCAL_COLUMNS]:
            require(parquet.schema_arrow.field(column).type == pa.int64(),
                    f"{side} metadata {column} must be int64")
        for column in ["source_file", "source_bbox", "attributes"]:
            require(parquet.schema_arrow.field(column).type == pa.string(),
                    f"{side} metadata {column} must be a string")
        allowed_sources = {s["path"] for s in lock["sources"] if s["side"] == side}
        tracker = SourceOrder(allowed_sources)
        offset = confidence_count = 0
        for batch in parquet.iter_batches(batch_size=batch_size):
            data = batch.to_pydict()
            n = len(batch)
            records = array[offset:offset + n]
            local = np.column_stack([data[k] for k in LOCAL_COLUMNS]).astype("<i8", copy=False)
            check_records(records, offset, local, shift)
            centers, have_confidence = check_metadata_rows(data, records, local, tracker, qa_pending[side])
            require(np.all(shapely.covers(projected_boundary, shapely.points(centers))),
                    "Retained polygon centroid is outside the fixed national boundary")
            confidence_count += have_confidence
            global_low = np.minimum(global_low, local[:, :2].min(axis=0))
            global_high = np.maximum(global_high, local[:, 2:].max(axis=0))
            max_coordinate = max(max_coordinate, int(records[:, 2:6].max()))
            for region in regions:
                bounds = region["window_projected_m"]
                inside = ((centers[:, 0] >= bounds[0]) & (centers[:, 0] < bounds[2]) &
                          (centers[:, 1] >= bounds[1]) & (centers[:, 1] < bounds[3]))
                if np.any(inside):
                    region[side].append(local[inside].copy())
            offset += n
            if progress and offset // 1_000_000 != (offset - n) // 1_000_000:
                print(f"Validated {side}: {offset:,}/{len(array):,}", flush=True)
        require(offset == len(array), "Incomplete metadata scan")
        require(not any(qa_pending[side].values()), "QA sample source record absent from published data")
        stats = info["filter_statistics"]
        excluded = sum(v for k, v in stats.items() if k.startswith(f"{side}.excluded."))
        require(stats[f"{side}.input_records"] == len(array) + excluded,
                f"{side} raw/retained/excluded accounting mismatch")
        require(stats[f"{side}.retained"] == len(array), f"{side} retained accounting mismatch")
        results[side] = {"records": len(array), "source_files": tracker.file_count,
                         "records_with_retained_confidence": confidence_count,
                         "source_record_uniqueness": "All source files occur in one block; record indexes and physical lines strictly increase within each source file",
                         "npy_sha256": file_sha256(array_path)}
    require([*map(int, global_low), *map(int, global_high)] == info["local_bounds_q1000"],
            "Manifest local bounds mismatch")
    require([int(global_low[0]) + shift[0], int(global_low[1]) + shift[1],
             int(global_high[0]) + shift[0], int(global_high[1]) + shift[1]] == info["global_bounds_q1000"],
            "Manifest shifted bounds mismatch")
    require(int(global_low[0]) + shift[0] == 1 and int(global_low[1]) + shift[1] == 1,
            "Common translation must map both global endpoint minima to one")
    groups = pq.read_table(path / "groups.parquet").to_pydict()
    require(len(groups["group_id"]) == 1 and groups["group_id"] == [0], "Invalid groups table")
    for key, expected in {"dx": shift[0], "dy": shift[1], "r1_start": 0, "r2_start": 0,
                          "r1_count": info["r1_count"], "r2_count": info["r2_count"],
                          "local_min_x": int(global_low[0]), "local_min_y": int(global_low[1]),
                          "local_max_x": int(global_high[0]), "local_max_y": int(global_high[1]),
                          "global_min_x": int(global_low[0]) + shift[0],
                          "global_max_x": int(global_high[0]) + shift[0]}.items():
        require(groups[key] == [expected], f"Groups table {key} mismatch")
    diagnostics = []
    for region in regions:
        local_arrays = []
        for side in ("r1", "r2"):
            chunks = region.pop(side)
            local_arrays.append(np.concatenate(chunks) if chunks else np.empty((0, 4), dtype="<i8"))
        aa, bb = local_arrays
        if progress:
            print(f"Checking fixed {region['name']} window: {len(aa):,} / {len(bb):,}", flush=True)
        diagnostics.append({**region, **region_join(aa, bb)})
    qa = qa_geometry(qa_path, projection, path / "qa_overlay.png")
    report = {
        "passed": True, "every_record_checked": True, "dataset": info["dataset"],
        "checks": ["NPY schema and sequential IDs", "unit weights and positive rectangles",
                   "one national group", "metadata row alignment", "finite source projected bounds",
                   "Q1000 ties-to-even endpoint quantization", "identical integer shift on both sides",
                   "source-file/source-record and physical-line uniqueness", "retained confidence metadata",
                   "all retained projected centroids covered by the fixed national boundary",
                   "source accounting", "manifest and group bounds", "original polygon projection QA"],
        "whole_population": results,
        "coordinate_offset": {"dx": shift[0], "dy": shift[1]},
        "numeric_range": {"max_coordinate": max_coordinate,
                          "note": "Rectangle and cumulative-mass arithmetic in each sampler still require its own overflow checks."},
        "local_diagnostics": {
            "scope": "Only the predeclared 10 km windows; not an estimate of national overlap count, density, or weight distributions.",
            "selection": "Select each side independently by projected polygon centroid inside a half-open window; intersect the full un-clipped selected rectangles. Cross-window-boundary matches to omitted records are not counted.",
            "does_not_filter_published_data": True, "national_pairs_enumerated": False,
            "pair_lists_saved": False, "regions": diagnostics,
        }, "polygon_qa": qa,
    }
    temporary_report = path / "validation.json.tmp"
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary_report.replace(path / "validation.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args()
    require(args.batch_size > 0, "batch-size must be positive")
    report = validate(args.dataset_dir, args.batch_size)
    print(json.dumps({"passed": report["passed"], "output": str(args.dataset_dir / "validation.json")}))


if __name__ == "__main__":
    main()
