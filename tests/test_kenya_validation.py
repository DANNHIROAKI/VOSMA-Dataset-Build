"""Regression tests for independent Kenya publication validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_kenya as validation


def have_modules(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


class RecordChecks(unittest.TestCase):
    def setUp(self):
        self.local = np.array([[-20, -10, 30, 40], [-20, -10, 30, 40]], dtype="<i8")
        self.records = np.array([[0, 0, 1, 1, 51, 51, 1], [1, 0, 1, 1, 51, 51, 1]], dtype="<i8")
        self.shift = (21, 11)

    def check(self):
        validation.check_records(self.records, 0, self.local, self.shift)

    def test_duplicate_geometry_with_distinct_ids_is_retained(self):
        self.check()

    def test_nonzero_chunk_offset(self):
        validation.check_records(self.records[1:], 1, self.local[1:], self.shift)

    def test_bad_record_id(self):
        self.records[1, 0] = 0
        with self.assertRaisesRegex(ValueError, "record_id"):
            self.check()

    def test_wrong_group(self):
        self.records[1, 1] = 1
        with self.assertRaisesRegex(ValueError, "group_id"):
            self.check()

    def test_nonunit_weight(self):
        self.records[1, 6] = 2
        with self.assertRaisesRegex(ValueError, "weight"):
            self.check()

    def test_wrong_translation(self):
        self.records[1, 2] = 2
        with self.assertRaisesRegex(ValueError, "common integer translation"):
            self.check()

    def test_nonpositive_area(self):
        self.records[0, 4] = 1
        with self.assertRaisesRegex(ValueError, "Nonpositive"):
            self.check()

    def test_nonpositive_coordinate(self):
        self.records[0, 2] = 0
        with self.assertRaisesRegex(ValueError, ">=1"):
            self.check()

    def test_overflow_is_rejected_before_int64_addition(self):
        self.local[0] = [validation.I64.max - 1, 0, validation.I64.max, 1]
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.check()


class SourceIdentityChecks(unittest.TestCase):
    def test_order_with_filtering_gaps(self):
        tracker = validation.SourceOrder()
        tracker.add("a.csv.gz", 0, 2)
        tracker.add("a.csv.gz", 100, 102)
        tracker.add("b.csv.gz", 0, 2)
        self.assertEqual(tracker.file_count, 2)
        self.assertEqual(tracker.records, 3)

    def test_duplicate_record(self):
        tracker = validation.SourceOrder()
        tracker.add("a", 1, 2)
        with self.assertRaisesRegex(ValueError, "source_record_index"):
            tracker.add("a", 1, 3)

    def test_duplicate_source_line(self):
        tracker = validation.SourceOrder()
        tracker.add("a", 1, 2)
        with self.assertRaisesRegex(ValueError, "physical source line"):
            tracker.add("a", 2, 2)

    def test_repeated_shard(self):
        tracker = validation.SourceOrder()
        tracker.add("a", 1, 2)
        tracker.add("b", 1, 2)
        with self.assertRaisesRegex(ValueError, "Repeated source-file block"):
            tracker.add("a", 2, 3)

    def test_source_must_belong_to_correct_side_of_lock(self):
        tracker = validation.SourceOrder({"microsoft.zip"})
        with self.assertRaisesRegex(ValueError, "source lock"):
            tracker.add("google.csv.gz", 1, 2)


class MetadataChecks(unittest.TestCase):
    def setUp(self):
        self.records = np.array([[0, 0, 1, 1, 3001, 4001, 1]], dtype="<i8")
        self.local = np.array([[-1000, -1000, 2000, 3000]], dtype="<i8")
        self.attrs = {"source_line_number": 2, "projected_centroid_m": [.5, 1],
                      "source_properties": {"confidence": ".8"}}
        self.data = {"record_id": [0], "group_id": [0], "source_file": ["source.csv.gz"],
                     "source_record_index": [0], "source_bbox": ["[-1,-1,2,3]"]}

    def check(self):
        self.data["attributes"] = [json.dumps(self.attrs)]
        return validation.check_metadata_rows(self.data, self.records, self.local, validation.SourceOrder())

    def test_valid_metadata(self):
        centers, confidence_count = self.check()
        self.assertEqual(centers.tolist(), [[.5, 1]])
        self.assertEqual(confidence_count, 1)

    def test_quantization_mismatch(self):
        self.data["source_bbox"] = ["[-1,-1,2,3.01]"]
        with self.assertRaisesRegex(ValueError, "quantization"):
            self.check()

    def test_centroid_outside_box(self):
        self.attrs["projected_centroid_m"] = [99, 99]
        with self.assertRaisesRegex(ValueError, "outside"):
            self.check()

    def test_nonfinite_confidence(self):
        self.attrs["source_properties"]["confidence"] = "nan"
        with self.assertRaisesRegex(ValueError, "confidence"):
            self.check()

    def test_qa_sample_is_linked_to_published_record(self):
        self.data["attributes"] = [json.dumps(self.attrs)]
        pending = {"source.csv.gz": {0: {"local_bbox_q1000": [-1000, -1000, 2000, 3001],
                                        "projected_bbox_m": [-1, -1, 2, 3],
                                        "projected_centroid_m": [.5, 1]}}}
        with self.assertRaisesRegex(ValueError, "QA sample bounds"):
            validation.check_metadata_rows(self.data, self.records, self.local,
                                           validation.SourceOrder(), pending)


class DistributionChecks(unittest.TestCase):
    def test_all_values_if_small(self):
        sample = validation.DistributionSample(capacity=10)
        sample.add(np.array([1., 3., 5.]), np.array([.2, .4, .6]))
        result = sample.finish()
        self.assertEqual(result["positive_pairs"], 3)
        self.assertEqual(result["quantile_sample_size"], 3)
        self.assertEqual(result["intersection_area_m2"]["median"], 3)
        self.assertEqual(result["iou"]["max"], .6)

    def test_bounded_sample_independent_of_batching(self):
        values = np.arange(1, 20001, dtype=float)
        one = validation.DistributionSample(capacity=20)
        many = validation.DistributionSample(capacity=20)
        one.add(values, values / len(values))
        for index in range(0, len(values), 131):
            part = values[index:index + 131]
            many.add(part, part / len(values))
        a, b = one.finish(), many.finish()
        for key in ("intersection_area_m2", "iou"):
            self.assertAlmostEqual(a[key].pop("mean"), b[key].pop("mean"))
        self.assertEqual(a, b)
        self.assertEqual(len(one.kept), 20)

    def test_empty_sample_json(self):
        result = validation.DistributionSample().finish()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["positive_pairs"], 0)


@unittest.skipUnless(have_modules("shapely"), "requires Shapely>=2")
class LocalOverlapChecks(unittest.TestCase):
    def test_boundary_contact_has_zero_mass(self):
        a = np.array([[0, 0, 1000, 1000], [2000, 2000, 3000, 3000]], dtype="<i8")
        b = np.array([[1000, 0, 2000, 1000], [500, 500, 1500, 1500]], dtype="<i8")
        result = validation.region_join(a, b)
        self.assertEqual(result["positive_pairs"], 1)
        self.assertEqual(result["r1_no_overlap"], 1)
        self.assertEqual(result["r2_no_overlap"], 1)
        self.assertEqual(result["intersection_area_m2"]["min"], .25)
        self.assertAlmostEqual(result["iou"]["min"], 1 / 7)

    def test_duplicate_geometry_multiplies_records(self):
        a = np.array([[0, 0, 1000, 1000]] * 3, dtype="<i8")
        b = a[:2]
        result = validation.region_join(a, b)
        self.assertEqual(result["positive_pairs"], 6)
        self.assertEqual(result["iou"]["min"], 1)

    def test_random_against_brute_force(self):
        rng = np.random.default_rng(20260922)
        for _ in range(30):
            boxes = []
            for n in (12, 15):
                lo = rng.integers(-10, 10, (n, 2))
                boxes.append(np.column_stack((lo, lo + rng.integers(1, 10, (n, 2)))))
            a, b = boxes
            count = sum(np.all(np.minimum(x[2:], y[2:]) > np.maximum(x[:2], y[:2])) for x in a for y in b)
            self.assertEqual(validation.region_join(a, b)["positive_pairs"], count)


@unittest.skipUnless(have_modules("pyarrow", "pyproj", "shapely", "matplotlib"), "requires geographic validation dependencies")
class FullValidationChecks(unittest.TestCase):
    def test_minimal_publication_with_duplicate_geometry(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyproj
        from shapely.geometry import Polygon, mapping
        from shapely.ops import transform

        projection = "+proj=laea +lat_0=0 +lon_0=37 +datum=WGS84 +units=m +no_defs"
        transformer = pyproj.Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        original = Polygon([(36.8218, -1.2922), (36.8220, -1.2922),
                            (36.8220, -1.2920), (36.8218, -1.2920)])
        projected = transform(transformer.transform, original)
        local = np.rint(np.asarray(projected.bounds) * validation.Q).astype("<i8")
        dx, dy = 1 - int(local[0]), 1 - int(local[1])
        shifted = local + [dx, dy, dx, dy]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            boundary = Polygon([(34, -5), (42, -5), (42, 5), (34, 5)])
            (path / "country_boundary.geojson").write_text(json.dumps(mapping(boundary)))
            lock = {"sources": [{"side": s, "path": s + ".geojsonl"} for s in ("r1", "r2")],
                    "boundary": {"sha256": validation.file_sha256(path / "country_boundary.geojson")}}
            (path / "SOURCE_LOCK.json").write_text(json.dumps(lock))
            samples = []
            for side in ("r1", "r2"):
                records = np.array([[index, 0, *shifted, 1] for index in range(2)], dtype="<i8")
                np.save(path / f"{side.upper()}.npy", records)
                rows = []
                for index in range(2):
                    props = {"side": side, "source_file": side + ".geojsonl", "source_record_index": index,
                             "local_bbox_q1000": local.tolist(), "projected_bbox_m": list(projected.bounds),
                             "projected_centroid_m": [projected.centroid.x, projected.centroid.y]}
                    attrs = {"source_line_number": index + 1, "projected_centroid_m": props["projected_centroid_m"],
                             "source_properties": {}}
                    rows.append({"record_id": index, "group_id": 0, "source_file": props["source_file"],
                                 "source_record_index": index, "source_bbox": json.dumps(list(projected.bounds)),
                                 "attributes": json.dumps(attrs), **dict(zip(validation.LOCAL_COLUMNS, local.tolist()))})
                    samples.append({"type": "Feature", "geometry": mapping(original), "properties": props})
                pq.write_table(pa.Table.from_pylist(rows), path / f"{side.upper()}_metadata.parquet")
            pq.write_table(pa.Table.from_pylist([{"group_id": 0, "dx": dx, "dy": dy,
                                                "r1_start": 0, "r2_start": 0, "r1_count": 2, "r2_count": 2,
                                                "local_min_x": int(local[0]), "local_min_y": int(local[1]),
                                                "local_max_x": int(local[2]), "local_max_y": int(local[3]),
                                                "global_min_x": int(shifted[0]), "global_max_x": int(shifted[2])}]),
                           path / "groups.parquet")
            info = {"dataset": "kenya_fixture", "quantization_scale": 1000, "group_count": 1,
                    "projection": projection, "coordinate_offset": {"dx": dx, "dy": dy},
                    "source_lock_sha256": validation.file_sha256(path / "SOURCE_LOCK.json"),
                    "local_bounds_q1000": local.tolist(), "global_bounds_q1000": shifted.tolist(),
                    "r1_count": 2, "r2_count": 2,
                    "filter_statistics": {"r1.input_records": 2, "r2.input_records": 2,
                                          "r1.retained": 2, "r2.retained": 2}}
            (path / "dataset.json").write_text(json.dumps(info))
            (path / "qa_samples.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": samples}))
            result = validation.validate(path, batch_size=1, progress=False)
            self.assertTrue(result["passed"])
            self.assertTrue((path / "qa_overlay.png").exists())
            self.assertEqual(result["local_diagnostics"]["regions"][0]["positive_pairs"], 4)
            self.assertEqual(result["local_diagnostics"]["regions"][1]["positive_pairs"], 0)
            self.assertTrue(result["local_diagnostics"]["does_not_filter_published_data"])
            self.assertFalse(result["local_diagnostics"]["pair_lists_saved"])


if __name__ == "__main__":
    unittest.main()
