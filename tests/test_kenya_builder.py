"""Small adversarial inputs exercise the national builder without network data."""
import csv
import gzip
import json
import pathlib
import zipfile

import numpy as np
import pyarrow.parquet as pq
import pytest
import shapely

from vosma_dataset import kenya


def polygon(x, y, width=0.001):
    return {"type": "Polygon", "coordinates": [[[x, y], [x + width, y],
            [x + width, y + width], [x, y + width], [x, y]]]}


@pytest.fixture
def fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    boundary = raw / "boundary.geojson"
    boundary.write_text(json.dumps({"type": "Polygon", "coordinates": [
        [[34, -5], [43, -5], [43, 6], [34, 6], [34, -5]]]}))
    p = polygon(36, 0)
    multi = {"type": "MultiPolygon", "coordinates": [polygon(36.2, 0)["coordinates"], polygon(36.3, 0)["coordinates"]]}
    invalid = {"type": "Polygon", "coordinates": [[[36, 0], [36.1, 0.1], [36, 0.1], [36.1, 0], [36, 0]]]}
    records = [p, p, multi, polygon(20, 0), invalid, {"type": "Point", "coordinates": [36, 0]}]
    microsoft = raw / "microsoft.zip"
    with zipfile.ZipFile(microsoft, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Kenya.geojsonl", "".join(json.dumps({"type": "Feature", "properties": {"confidence": 0}, "geometry": g}) + "\n" for g in records) + "not-json\n")
    sources = [{"side": "r1", "format": "geojsonl.zip", "path": microsoft.name,
                "sha256": kenya.sha256(microsoft), "url": "https://example.org/microsoft.zip", "expected_records": 7}]
    for number, boxes in enumerate(([p, p, polygon(20, 0)], [polygon(37.5, 0.01)])):
        google = raw / f"google-{number}.csv.gz"
        with gzip.open(google, "wt", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["geometry", "confidence", "full_plus_code"])
            for index, geometry in enumerate(boxes):
                writer.writerow([shapely.to_wkt(shapely.from_geojson(json.dumps(geometry))), "0" if index == 0 else "0.999", "duplicate-id"])
        sources.append({"side": "r2", "format": "csv.gz", "path": google.name,
                        "sha256": kenya.sha256(google), "url": f"https://example.org/{google.name}", "expected_records": len(boxes)})
    lock = tmp_path / "sources.lock.json"
    lock.write_text(json.dumps({"dataset": "kenya_buildings", "boundary": {"path": boundary.name,
                               "sha256": kenya.sha256(boundary), "url": "https://example.org/boundary.geojson"}, "sources": sources}))
    return raw, lock, tmp_path / "out", tmp_path / "work"


def run_fixture(fixture):
    return kenya.build(*fixture, batch_size=2, shard_records=3, qa_samples=2)


def test_national_geometry_and_resume(fixture):
    output = run_fixture(fixture)
    info = json.loads((output / "dataset.json").read_text())
    assert (info["r1_count"], info["r2_count"]) == (3, 3)
    assert info["group_count"] == 1
    assert info["filter_statistics"]["r1.input_records"] == 7
    assert info["filter_statistics"]["r1.excluded.invalid_geometry"] == 1
    assert info["filter_statistics"]["r1.excluded.centroid_outside_country"] == 1
    assert info["filter_statistics"]["r1.excluded.unsupported_geometry_type"] == 1
    assert info["filter_statistics"]["r1.excluded.malformed_record"] == 1
    assert info["filter_statistics"]["r2.excluded.centroid_outside_country"] == 1
    dx, dy = info["coordinate_offset"]["dx"], info["coordinate_offset"]["dy"]
    assert dx > 0  # Unshifted LAEA x is negative west of its fixed centre.
    for side in ("R1", "R2"):
        array = np.load(output / (side + ".npy"), mmap_mode="r")
        assert array.dtype.str == "<i8"
        assert array.shape == (3, 7)
        assert np.array_equal(array[:, 0], np.arange(3))
        assert np.all(array[:, 1] == 0) and np.all(array[:, 6] == 1)
        meta = pq.read_table(output / (side + "_metadata.parquet")).to_pylist()
        for row, md in zip(array, meta):
            local = np.array([md[k] for k in ("local_x0", "local_y0", "local_x1", "local_y1")])
            assert np.array_equal(row[2:6], local + [dx, dy, dx, dy])
            assert np.array_equal(kenya.quantized_bounds(json.loads(md["source_bbox"])), local)
        # Identical records are retained, irrespective of IDs/confidence.
        assert np.array_equal(array[0, 2:6], array[1, 2:6])
    assert min(np.load(output / (side + ".npy"))[:, 2:6].min() for side in ("R1", "R2")) == 1
    r1meta = pq.read_table(output / "R1_metadata.parquet").to_pylist()
    multi = json.loads(r1meta[2]["source_bbox"])
    assert multi[2] - multi[0] > 10000  # Full MultiPolygon bounds, one source record.
    r2meta = pq.read_table(output / "R2_metadata.parquet").to_pylist()
    assert [row["source_record_index"] for row in r2meta] == [1, 2, 1]
    assert json.loads(r2meta[0]["attributes"])["source_line_number"] == 2
    with gzip.open(output / "exclusions.jsonl.gz", "rt") as stream:
        assert len([json.loads(line) for line in stream]) == 5
    before = (output / "SHA256SUMS").read_bytes()
    assert run_fixture(fixture) == output
    assert (output / "SHA256SUMS").read_bytes() == before


def test_resume_rejects_corrupted_checkpoint(fixture):
    run_fixture(fixture)
    path = next(fixture[3].glob("source-*/part-*/rectangles.i64"))
    with path.open("r+b") as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        run_fixture(fixture)


def test_source_checksum_prevents_reusing_changed_inputs(fixture):
    raw, lock, out, work = fixture
    with (raw / "microsoft.zip").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        run_fixture(fixture)


def test_ties_to_even_quantization():
    assert kenya.quantized_bounds([-0.0025, -0.0015, 0.0025, 0.0035]).tolist() == [-2, -2, 2, 4]


def test_no_output_overwrite(fixture):
    output = fixture[2] / "kenya_buildings"
    output.mkdir(parents=True)
    (output / "sentinel").write_text("existing user data")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_fixture(fixture)
    assert (output / "sentinel").read_text() == "existing user data"


def test_serial_parallel_byte_identity(fixture):
    serial = run_fixture(fixture)
    raw, lock, output, work = fixture
    parallel = kenya.build(raw, lock, output.with_name("parallel-out"), work.with_name("parallel-work"),
                           batch_size=2, shard_records=3, qa_samples=2, workers=2)
    assert (serial / "SHA256SUMS").read_bytes() == (parallel / "SHA256SUMS").read_bytes()
