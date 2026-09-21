"""Independent adversarial tests for selection and original-footprint QA."""
import csv
import gzip
import json
import zipfile

import numpy as np
import pyproj
import pytest
import shapely

import qa_kenya_local_overlay as overlay
import write_kenya_release_docs as release_docs
from vosma_dataset import kenya


@pytest.fixture
def local_dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    boundary = raw / "country.geojson"
    boundary.write_text(json.dumps({"type": "Polygon", "coordinates": [
        [[34, -5], [43, -5], [43, 6], [34, 6], [34, -5]]]}))
    forward = kenya.make_transformer()
    inverse = pyproj.Transformer.from_crs(forward.target_crs, "EPSG:4326", always_xy=True)
    center = forward.transform(36.8219, -1.2921)

    def original_rectangle(x_offset):
        x, y = center[0] + x_offset, center[1]
        ring = [inverse.transform(a, b) for a, b in
                [(x - 4, y - 4), (x + 4, y - 4), (x + 4, y + 4),
                 (x - 4, y + 4), (x - 4, y - 4)]]
        return {"type": "Polygon", "coordinates": [ring]}

    # R1 IDs 0/1 are duplicate geometries but distinct source records. R2 ID 2
    # matches them while its smaller IDs do not; IoU-based selection would fail.
    r1_geometries = [original_rectangle(x) for x in (-50, -50, 10, 140)]
    r2_geometries = [original_rectangle(x) for x in (50, 60, -50, -140)]
    r1_records = [{"type": "Feature", "id": str(index),
                   "properties": {"source_tail_guard": "do-not-parse"} if index >= 2 else {"label": "selected"},
                   "geometry": geometry}
                  for index, geometry in enumerate(r1_geometries)]
    microsoft = raw / "microsoft.zip"
    with zipfile.ZipFile(microsoft, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Kenya.geojsonl", "".join(json.dumps(row) + "\n" for row in r1_records))
    google = raw / "google.csv.gz"
    with gzip.open(google, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["geometry", "confidence", "full_plus_code"])
        for index, geometry in enumerate(r2_geometries):
            writer.writerow([shapely.to_wkt(shapely.from_geojson(json.dumps(geometry)), rounding_precision=-1),
                             "0.65", f"building-{index}"])
    sources = [{"side": side, "format": fmt, "path": path.name,
                "url": f"https://fixture.invalid/{path.name}", "sha256": kenya.sha256(path),
                "expected_records": 4}
               for side, fmt, path in [("r1", "geojsonl.zip", microsoft), ("r2", "csv.gz", google)]]
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"dataset": "kenya_buildings", "boundary": {
        "path": boundary.name, "url": "https://fixture.invalid/country.geojson",
        "sha256": kenya.sha256(boundary)}, "sources": sources}))
    dataset = kenya.build(raw, lock, tmp_path / "release", tmp_path / "checkpoints",
                          batch_size=2, shard_records=3, qa_samples=1)
    return dataset, raw, r1_records


def by_name(report):
    return {region["name"]: region for region in report["regions"]}


def test_lowest_ids_duplicates_empty_windows_and_no_overlap_selection(local_dataset):
    dataset, raw, _ = local_dataset
    report = overlay.build_overlay(dataset, raw, limit=2)
    assert report["passed"] is True
    regions = by_name(report)
    assert set(regions) == {"Nairobi", "Kisumu", "Turkana"}
    nairobi = regions["Nairobi"]
    for side in ("r1", "r2"):
        assert nairobi[side]["selected_count"] == 2
        assert nairobi[side]["record_ids"] == [0, 1]
        assert [row["source_record_index"] for row in nairobi[side]["source_records"]] == [1, 2]
        assert [row["record_id"] for row in nairobi[side]["source_records"]] == [0, 1]
        array = np.load(dataset / (side.upper() + ".npy"), mmap_mode="r")
        for md in nairobi[side]["source_records"]:
            assert md["published_bbox_q1000"] == array[md["record_id"], 2:6].tolist()
    r1 = nairobi["r1"]["source_records"]
    r2 = nairobi["r2"]["source_records"]
    assert r1[0]["published_bbox_q1000"] == r1[1]["published_bbox_q1000"]
    assert all(a["published_bbox_q1000"][2] < b["published_bbox_q1000"][0] for a in r1 for b in r2)
    for name in ("Kisumu", "Turkana"):
        assert regions[name]["r1"]["selected_count"] == 0
        assert regions[name]["r2"]["selected_count"] == 0
        assert regions[name]["r1"]["record_ids"] == []
        assert regions[name]["r2"]["record_ids"] == []
    for scan in report["source_scans"]:
        assert scan["max_requested_record_index"] == 2
        assert scan["records_scanned"] == 2
    assert (dataset / "qa_local_overlay.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    saved = json.loads((dataset / "qa_local_overlay.json").read_text())
    assert saved["passed"] is True
    assert by_name(saved)["Nairobi"]["r1"]["record_ids"] == [0, 1]
    info = json.loads((dataset / "dataset.json").read_text())
    assert release_docs.validate_local_overlay(dataset, info) == saved


def test_unselected_json_tail_is_not_parsed(local_dataset, monkeypatch):
    dataset, raw, _ = local_dataset
    original_loads = json.loads
    def guard(text, *args, **kwargs):
        if isinstance(text, str) and '"type": "Feature"' in text and "source_tail_guard" in text:
            raise AssertionError("Unselected raw source tail must not be parsed")
        return original_loads(text, *args, **kwargs)
    monkeypatch.setattr(json, "loads", guard)
    report = overlay.build_overlay(dataset, raw, limit=2)
    assert report["passed"] is True
    assert all(scan["records_scanned"] == 2 for scan in report["source_scans"])


def test_modified_selected_npy_rectangle_is_rejected(local_dataset):
    dataset, raw, _ = local_dataset
    array = np.load(dataset / "R1.npy", mmap_mode="r+")
    array[0, 2] += 1
    array.flush()
    del array
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        overlay.build_overlay(dataset, raw, limit=2)


@pytest.fixture
def documented_local_qa(local_dataset, monkeypatch):
    dataset, raw, _ = local_dataset
    # Selection, original-footprint verification and report generation stay real.
    # The two earlier tests exercise actual rendering; only redundant drawing is
    # replaced here to keep the publication-gate fault matrix inexpensive.
    def draw_fixture_png(selected, regions, output_path):
        output_path.write_bytes(b"\x89PNG\r\n\x1a\npublication-gate-fixture")
    monkeypatch.setattr(overlay, "draw_panels", draw_fixture_png)
    overlay.build_overlay(dataset, raw, limit=2)
    return dataset, json.loads((dataset / "dataset.json").read_text())


@pytest.mark.parametrize("change", [
    "png", "source_lock", "r1_count", "r2_count", "offset",
    "geographic_center", "shifted_projected_window", "unsorted_ids",
])
def test_publication_gate_rejects_changed_local_qa(documented_local_qa, change):
    dataset, info = documented_local_qa
    path = dataset / "qa_local_overlay.json"
    report = json.loads(path.read_text())
    if change == "png":
        with (dataset / "qa_local_overlay.png").open("ab") as stream:
            stream.write(b"tampered after verification")
    elif change == "source_lock":
        report["source_lock_sha256"] = "0" * 64
    elif change in ("r1_count", "r2_count"):
        report[change] += 1
    elif change == "offset":
        report["coordinate_offset"]["dx"] += 1
    elif change == "geographic_center":
        report["regions"][0]["center_wgs84_lon_lat"][0] += 0.01
    elif change == "shifted_projected_window":
        # Keeping width=height=200 and the geographic label unchanged must not
        # allow an independently shifted projected window to pass publication.
        region = report["regions"][0]
        region["center_projected_m"][0] += 250
        region["window_projected_m"][0] += 250
        region["window_projected_m"][2] += 250
    elif change == "unsorted_ids":
        sample = report["regions"][0]["r1"]
        sample["record_ids"].reverse()
        sample["source_records"].reverse()
    path.write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError):
        release_docs.validate_local_overlay(dataset, info)


@pytest.mark.parametrize("change", ["geometry", "properties"])
def test_modified_selected_original_source_is_rejected(local_dataset, change):
    dataset, raw, records = local_dataset
    if change == "geometry":
        records[0]["geometry"]["coordinates"][0] = [
            [x + 0.0001, y] for x, y in records[0]["geometry"]["coordinates"][0]]
    else:
        records[0]["properties"]["label"] = "tampered selected source metadata"
    with zipfile.ZipFile(raw / "microsoft.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Kenya.geojsonl", "".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        overlay.build_overlay(dataset, raw, limit=2)
