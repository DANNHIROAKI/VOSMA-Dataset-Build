#!/usr/bin/env python3
"""Recheck and overlay raw footprints in three fixed 200 m geographic windows.

This is supplementary visual QA, not a population-quality or overlap estimate.
Selection is deterministic by published record ID, independently on each side.
Neither intersection scores nor visual appearance participate in selection.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pyarrow.parquet as pq
import pyproj
import shapely
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry

REGIONS = (("Nairobi", 36.8219, -1.2921),
           ("Kisumu", 34.7617, -0.0917),
           ("Turkana", 35.6000, 3.1000))
HALF_SIZE_M = 100
Q = 1000
LOCAL_COLUMNS = ["local_x0", "local_y0", "local_x1", "local_y1"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_path(raw_root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source path")
    result = (raw_root / relative).resolve()
    require(result.is_relative_to(raw_root.resolve()), "Source path escapes raw root")
    return result


def select_records(dataset_dir, info, regions, limit):
    """Streaming metadata scan; JSON is decoded only near an unsaturated window."""
    targets = {}
    selected = {name: {side: [] for side in ("r1", "r2")} for name, _, _ in REGIONS}
    scan_counts = {}
    shift = info["coordinate_offset"]
    translated = np.array([shift["dx"], shift["dy"], shift["dx"], shift["dy"]], dtype="<i8")
    columns = ["record_id", "group_id", "source_file", "source_record_index",
               "source_annotation_id", "source_bbox", "attributes", *LOCAL_COLUMNS]
    for side in ("r1", "r2"):
        array = np.load(dataset_dir / f"{side.upper()}.npy", mmap_mode="r", allow_pickle=False)
        require(array.dtype == np.dtype("<i8") and array.shape == (info[f"{side}_count"], 7),
                f"{side} array shape/type mismatch")
        parquet = pq.ParquetFile(dataset_dir / f"{side.upper()}_metadata.parquet")
        require(parquet.metadata.num_rows == len(array), "Metadata/array row counts differ")
        offset = 0
        for batch in parquet.iter_batches(batch_size=50000, columns=columns):
            data = batch.to_pydict()
            ids = np.asarray(data["record_id"], dtype="<i8")
            require(np.array_equal(ids, np.arange(offset, offset + len(batch))),
                    "Metadata is not in increasing published record-ID order")
            local = np.column_stack([data[k] for k in LOCAL_COLUMNS]).astype("<i8", copy=False)
            decoded = {}
            for region in regions:
                records = selected[region["name"]][side]
                if len(records) == limit:
                    continue
                x0, y0, x1, y1 = region["window_projected_m"]
                # A polygon's centroid is within its unquantized bounding box.
                # One integer unit of padding conservatively absorbs endpoint rounding.
                candidates = np.flatnonzero((local[:, 2] >= x0 * Q - 1) &
                                            (local[:, 0] <= x1 * Q + 1) &
                                            (local[:, 3] >= y0 * Q - 1) &
                                            (local[:, 1] <= y1 * Q + 1))
                for row in candidates:
                    attrs = decoded.setdefault(int(row), None)
                    if attrs is None:
                        attrs = json.loads(data["attributes"][row])
                        decoded[int(row)] = attrs
                    cx, cy = attrs["projected_centroid_m"]
                    require(np.isfinite([cx, cy]).all(), "Nonfinite selected centroid metadata")
                    if not (x0 <= cx < x1 and y0 <= cy < y1):
                        continue
                    rid = int(ids[row])
                    output = array[rid]
                    require(int(output[0]) == rid and int(output[1]) == data["group_id"][row] == 0,
                            "Selected record identity/group mismatch")
                    require(output[6] == 1 and np.all(output[2:6] >= 1) and
                            np.all(output[2:4] < output[4:6]), "Invalid selected published rectangle")
                    require(np.array_equal(local[row] + translated, output[2:6]),
                            "Selected published rectangle differs from common-shift metadata")
                    record = {"region": region["name"], "side": side, "record_id": rid,
                              "source_file": data["source_file"][row],
                              "source_record_index": data["source_record_index"][row],
                              "source_line_number": attrs["source_line_number"],
                              "source_annotation_id": data["source_annotation_id"][row],
                              "local_bbox_q1000": local[row].tolist(),
                              "published_bbox_q1000": output[2:6].tolist(),
                              "projected_bbox_m": json.loads(data["source_bbox"][row]),
                              "projected_centroid_m": [cx, cy],
                              "_window_projected_m": region["window_projected_m"],
                              "_source_properties": attrs["source_properties"]}
                    key = (side, record["source_file"], record["source_record_index"])
                    require(key not in targets, "Source record was selected more than once")
                    targets[key] = record
                    records.append(record)
                    if len(records) == limit:
                        break
            offset += len(batch)
            if all(len(selected[region["name"]][side]) == limit for region in regions):
                break
        scan_counts[side] = offset
    return selected, targets, scan_counts


def verify_original(record, geometry, properties, annotation_id, physical_line, transformer):
    require(physical_line == record["source_line_number"], "Original physical source line mismatch")
    require(properties == record["_source_properties"], "Original source properties mismatch")
    require(str(annotation_id) == record["source_annotation_id"], "Original source annotation ID mismatch")
    require(geometry.geom_type in ("Polygon", "MultiPolygon") and not geometry.is_empty and
            geometry.is_valid and np.isfinite(shapely.get_coordinates(geometry)).all(),
            "Selected original polygon is invalid")
    projected = transform_geometry(transformer.transform, geometry)
    require(projected.is_valid and not projected.is_empty and projected.area > 0 and
            np.isfinite(shapely.get_coordinates(projected)).all(), "Selected projected polygon is invalid")
    bounds = np.asarray(projected.bounds)
    quantized = np.rint(bounds * Q).astype("<i8")
    require(np.array_equal(quantized, record["local_bbox_q1000"]),
            "Raw polygon quantized bounds differ from published rectangle")
    require(np.allclose(bounds, record["projected_bbox_m"], atol=1e-6, rtol=0),
            "Raw polygon projected bounds differ from metadata")
    centroid = [projected.centroid.x, projected.centroid.y]
    require(np.allclose(centroid, record["projected_centroid_m"], atol=1e-6, rtol=0),
            "Raw polygon projected centroid differs from metadata")
    x0, y0, x1, y1 = record["_window_projected_m"]
    require(x0 <= centroid[0] < x1 and y0 <= centroid[1] < y1,
            "Recomputed original polygon centroid lies outside its fixed window")
    record["_projected_geometry"] = projected


def scan_source(stream, source, wanted, transformer):
    """Read only up to the last required source record, without parsing other geometry."""
    maximum = max(wanted)
    scanned = physical_line = found = 0
    if source["format"].startswith("geojsonl"):
        for index, line in enumerate(stream, 1):
            scanned = physical_line = index
            if index in wanted:
                value = json.loads(line)
                is_feature = value.get("type") == "Feature"
                geom = shape(value["geometry"] if is_feature else value)
                properties = (value.get("properties") or {}) if is_feature else {}
                annotation_id = str(value.get("id", "")) if is_feature else ""
                verify_original(wanted[index], geom, properties, annotation_id, index, transformer)
                found += 1
            if index == maximum:
                break
    else:
        csv.field_size_limit(64 * 1024 * 1024)
        reader = csv.DictReader(stream, strict=True)
        geometry_column = source.get("geometry_column", "geometry")
        require(reader.fieldnames and geometry_column in reader.fieldnames, "Missing source WKT column")
        for index, row in enumerate(reader, 1):
            scanned, physical_line = index, reader.line_num
            if index in wanted:
                require(None not in row and all(value is not None for value in row.values()),
                        "Selected original CSV row is malformed")
                geom = shapely.from_wkt(row[geometry_column])
                properties = {key: value for key, value in row.items() if key != geometry_column}
                verify_original(wanted[index], geom, properties, str(row.get("full_plus_code", "")),
                                physical_line, transformer)
                found += 1
            if index == maximum:
                break
    require(scanned == maximum and found == len(wanted), "Required original source records were not found")
    return {"source_file": source["path"], "side": source["side"],
            "max_requested_record_index": maximum, "records_scanned": scanned,
            "physical_lines_scanned": physical_line, "selected_source_records": found}


def fetch_originals(raw_root, lock, targets, transformer):
    sources = {(source["side"], source["path"]): source for source in lock["sources"]}
    per_source = {}
    for (side, name, index), record in targets.items():
        require((side, name) in sources, "Selected source does not belong to the frozen source lock")
        require(isinstance(index, int) and index >= 1, "Expected one-based original source record index")
        per_source.setdefault((side, name), {})[index] = record
    scans = []
    for key, wanted in per_source.items():
        source = sources[key]
        path = raw_path(raw_root, source["path"])
        if "size_bytes" in source:
            require(path.stat().st_size == source["size_bytes"], "Selected source archive size differs from lock")
        print(json.dumps({"phase": "scan_original_source", "path": source["path"],
                          "through_record": max(wanted), "selected_records": len(wanted)}), flush=True)
        if source["format"] == "geojsonl.zip":
            with zipfile.ZipFile(path) as archive:
                names = [info.filename for info in archive.infolist() if not info.is_dir()]
                member = source.get("member")
                if member is None:
                    require(len(names) == 1, "ZIP source requires an explicit member")
                    member = names[0]
                require(member in names, "Frozen source ZIP member is missing")
                with archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                    scans.append(scan_source(text, source, wanted, transformer))
        elif source["format"] == "csv.gz":
            with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as text:
                scans.append(scan_source(text, source, wanted, transformer))
        else:
            require(source["format"] in ("geojsonl", "csv"), "Unsupported locked source format")
            with path.open(encoding="utf-8-sig", newline="") as text:
                scans.append(scan_source(text, source, wanted, transformer))
    return scans


def draw_panels(selected, regions, output_path):
    with tempfile.TemporaryDirectory(prefix=".local-qa-cache-", dir=output_path.parent) as cache:
        previous = {name: os.environ.get(name) for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
        os.environ["MPLCONFIGDIR"] = str(Path(cache) / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(Path(cache) / "xdg")
        try:
            _draw_panels(selected, regions, output_path)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _draw_panels(selected, regions, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as PlotPath
    from shapely.geometry.polygon import orient

    colors = {"r1": "#2563eb", "r2": "#e06b22"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7))
    for ax, region in zip(axes, regions):
        cx, cy = region["center_projected_m"]
        for side in ("r1", "r2"):
            for record in selected[region["name"]][side]:
                geometry = record["_projected_geometry"]
                for polygon in geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]:
                    polygon = orient(polygon, sign=1)
                    vertices, codes = [], []
                    for ring in [polygon.exterior, *polygon.interiors]:
                        points = np.asarray(ring.coords)[:, :2] - [cx, cy]
                        vertices.extend(points.tolist())
                        codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 2) + [PlotPath.CLOSEPOLY])
                    ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=colors[side],
                                           edgecolor=colors[side], linewidth=1, alpha=.35))
                x0, y0, x1, y1 = np.asarray(record["local_bbox_q1000"], dtype=float) / Q
                ax.add_patch(Rectangle((x0 - cx, y0 - cy), x1 - x0, y1 - y0,
                                       facecolor="none", edgecolor=colors[side], linewidth=.95,
                                       linestyle="--", alpha=.95))
        counts = {side: len(selected[region["name"]][side]) for side in ("r1", "r2")}
        ax.set_title(f"{region['name']}\nR1: {counts['r1']}  |  R2: {counts['r2']}", fontsize=11)
        if not sum(counts.values()):
            ax.text(.5, .5, "No records in this fixed window", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#64748b")
        ax.set_xlim(-HALF_SIZE_M, HALF_SIZE_M)
        ax.set_ylim(-HALF_SIZE_M, HALF_SIZE_M)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East of fixed center (m)")
        ax.set_ylabel("North of fixed center (m)")
        ax.grid(alpha=.15)
    handles = []
    for side, source in (("r1", "Microsoft"), ("r2", "Google")):
        handles.extend([Line2D([0], [0], color=colors[side], linewidth=1.4, label=f"{source} original footprint"),
                        Line2D([0], [0], color=colors[side], linestyle="--", label=f"{source} published rectangle")])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Fixed 200 m windows: original footprints and published rectangles", fontsize=14)
    fig.subplots_adjust(left=.065, right=.985, bottom=.17, top=.82, wspace=.28)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_overlay(dataset_dir, raw_root, limit=12):
    require(isinstance(limit, int) and 1 <= limit <= 12, "Sample limit must be between 1 and 12")
    dataset_dir, raw_root = Path(dataset_dir), Path(raw_root)
    info = json.loads((dataset_dir / "dataset.json").read_text())
    require(info["dataset"] == "kenya_buildings" and info["quantization_scale"] == Q and
            info["group_count"] == 1, "Expected the Q1000 single-group Kenya dataset")
    lock_path = dataset_dir / "SOURCE_LOCK.json"
    require(sha256(lock_path) == info["source_lock_sha256"], "Source lock digest mismatch")
    lock = json.loads(lock_path.read_text())
    projection = info["projection"]
    if not any(option in projection for option in ("+datum=", "+ellps=", "+R=")):
        require(info.get("projection_datum") == "WGS84", "Expected the explicit WGS84 datum")
        projection += " +datum=WGS84"
    transformer = pyproj.Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    regions = []
    for name, lon, lat in REGIONS:
        cx, cy = transformer.transform(lon, lat)
        regions.append({"name": name, "center_wgs84_lon_lat": [lon, lat],
                        "center_projected_m": [cx, cy],
                        "window_projected_m": [cx - HALF_SIZE_M, cy - HALF_SIZE_M,
                                                cx + HALF_SIZE_M, cy + HALF_SIZE_M]})
    selected, targets, scan_counts = select_records(dataset_dir, info, regions, limit)
    scans = fetch_originals(raw_root, lock, targets, transformer)
    output_path = dataset_dir / "qa_local_overlay.png"
    draw_panels(selected, regions, output_path)
    for region in regions:
        for side in ("r1", "r2"):
            records = selected[region["name"]][side]
            region[side] = {"selected_count": len(records), "record_ids": [r["record_id"] for r in records],
                            "source_records": [{key: value for key, value in r.items() if not key.startswith("_")}
                                               for r in records]}
    report = {
        "passed": True, "dataset": "kenya_buildings", "window_size_m": 2 * HALF_SIZE_M,
        "r1_count": info["r1_count"], "r2_count": info["r2_count"],
        "maximum_records_per_side_per_window": limit, "original_polygons_checked": len(targets),
        "selection": "Each fixed half-open projected 200 m square independently selects up to the stated limit from each side by ascending published record_id, requiring the original projected polygon centroid to lie inside. No IoU, intersection, matching or appearance criterion is used.",
        "scope": "Deterministic local visual QA only; not an estimate of national quality, overlap, density or weight distributions. Empty windows remain empty and are never relocated.",
        "visual_crop": "Complete original projected geometries and complete Q1000 rectangles are drawn at their true common relative positions; panel axes visually crop to the fixed window. No data geometry is clipped or altered.",
        "published_coordinates": "Rectangle corners are checked against the final NPY arrays after the dataset's common integer shift; displayed coordinates remove that same shift and subtract the fixed panel center in metres.",
        "projection": projection, "coordinate_offset": info["coordinate_offset"],
        "source_lock_sha256": info["source_lock_sha256"],
        "raw_source_verification": "Selected records are re-read from the frozen source files and checked against metadata and final arrays. Archive sizes are checked when available. The main build verifies full archive SHA256; this supplementary scan stops after each required maximum record rather than rehashing or reparsing complete archives.",
        "metadata_rows_scanned": scan_counts, "source_scans": scans, "regions": regions,
        "overlay": output_path.name, "overlay_sha256": sha256(output_path),
        "does_not_filter_published_data": True, "pair_lists_saved": False,
    }
    temporary = dataset_dir / "qa_local_overlay.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(dataset_dir / "qa_local_overlay.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    report = build_overlay(args.dataset_dir, args.raw_root, args.limit)
    print(json.dumps({"passed": report["passed"], "original_polygons_checked": report["original_polygons_checked"],
                      "overlay": str(args.dataset_dir / report["overlay"])}), flush=True)


if __name__ == "__main__":
    main()
