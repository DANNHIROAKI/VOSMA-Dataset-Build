import gzip
import json
import pathlib
import re
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pyarrow.parquet as pq

from vosma_dataset.doc_prediction import build
from vosma_dataset.common import sha256


CATEGORY_NAMES = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title",
]


def exact_json(value):
    """Write Decimal coordinates as JSON numbers, without binary float conversion."""
    text = json.dumps(value, default=lambda item: "__decimal__" + str(item))
    return re.sub(r'"__decimal__([^"\\]+)"', r'\1', text)


class DocPredictionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="doc-predictions-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        self.raw, self.out = self.root / "raw", self.root / "out"
        (self.raw / "doclaynet" / "COCO").mkdir(parents=True)
        self.pages_path = self.root / "pages.json"
        self.predictions_path = self.root / "predictions.jsonl.gz"
        self.protocol_path = self.root / "protocol.json"
        self.protocol = {
            "schema_version": "doclaynet-predictions-v1",
            "model_id": "fixed-test-model", "revision": "0123456789abcdef",
            "model_sha256": "b" * 64, "source_page_splits": ["test"],
            "device": "cpu", "dtype": "float32",
            "ground_truth_used_for_prediction": False,
            "id2label": {"0": "N/A", **{str(index): name
                         for index, name in enumerate(CATEGORY_NAMES, 1)}},
            "postprocessing": {
                "top_k": 100, "confidence_threshold": 0.7, "threshold_operator": ">",
                "excluded_label_ids": [0], "nms": False,
                "clipping": False, "deduplication": False,
            },
        }
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.page = {
            "page_id": "test/1", "split": "test", "source_image_id": 1,
            "file_name": "page1.png", "width": 1025, "height": 1025,
            "physical_key": ["scientific_articles", "collection", "document", 1],
            "image_sha256": "a" * 64,
        }

    @staticmethod
    def prediction(rank=0, label=11, box=(-1, -2, 2, 2), query=7):
        return {
            "prediction_index": rank, "query_index": query, "label": label,
            "score": Decimal("0.9000000357627869"),
            **dict(zip(("x0", "y0", "x1", "y1"), box)),
        }

    def result(self, predictions=(), **updates):
        result = {
            "page_id": "test/1", "status": "success", "image_sha256": "a" * 64,
            "width": 1025, "height": 1025, "model_input_shape": [1, 3, 800, 800],
            "top_k_candidates": 100, "threshold_excluded": 100 - len(predictions),
            "background_excluded": 0, "nonfinite_score_excluded": 0,
            "predictions": list(predictions),
        }
        result.update(updates)
        return result

    @staticmethod
    def annotation(image_id=1, annotation_id=1, category=1, bbox=(-2, -3, 4, 6)):
        return {
            "id": annotation_id, "image_id": image_id,
            "category_id": category, "bbox": list(bbox), "precedence": 0,
        }

    def write_inputs(self, results, annotations=(), pages=None, extra_images=()):
        pages = [self.page] if pages is None else pages
        self.pages_path.write_text(exact_json({
            "schema_version": "doclaynet-pages-v1", "splits": ["test"], "pages": pages,
        }))
        self.protocol["page_manifest_sha256"] = sha256(self.pages_path)
        self.protocol_path.write_text(json.dumps(self.protocol))
        protocol_digest = sha256(self.protocol_path)
        images = []
        for page in pages:
            images.append({
                "id": page["source_image_id"], "file_name": page["file_name"],
                "width": page["width"], "height": page["height"],
                **dict(zip(("doc_category", "collection", "doc_name", "page_no"),
                           page["physical_key"])),
            })
        images.extend(extra_images)
        (self.raw / "doclaynet" / "COCO" / "test.json").write_text(exact_json({
            "images": images, "annotations": list(annotations),
            "categories": [{"id": index, "name": name}
                           for index, name in enumerate(CATEGORY_NAMES, 1)],
        }))
        with gzip.open(self.predictions_path, "wt") as stream:
            for result in results:
                result.setdefault("protocol_sha256", protocol_digest)
                stream.write(exact_json(result) + "\n")

    def resign_protocol(self):
        """Update the fixture hash chain after a deliberate protocol edit."""
        self.protocol_path.write_text(json.dumps(self.protocol))
        protocol_digest = sha256(self.protocol_path)
        with gzip.open(self.predictions_path, "rt") as stream:
            results = [json.loads(line, parse_float=Decimal) for line in stream]
        with gzip.open(self.predictions_path, "wt") as stream:
            for result in results:
                result["protocol_sha256"] = protocol_digest
                stream.write(exact_json(result) + "\n")

    def build(self):
        return build(self.raw, self.out, self.pages_path,
                     self.predictions_path, self.protocol_path)

    def test_successful_page_with_zero_predictions_is_retained(self):
        self.write_inputs([self.result()], annotations=[])
        report = self.build()
        self.assertEqual(report["group_count"], 1)
        self.assertEqual(report["r1_count"], 0)
        self.assertEqual(report["r2_count"], 0)
        self.assertEqual(report["group_states"], {"both_empty": 1})
        self.assertEqual(report["inference_counts"]["zero_output_successful_pages"], 1)
        self.assertEqual(report["inference_counts"]["threshold_excluded"], 100)
        self.assertEqual(report["model_protocol"], self.protocol)

    def test_missing_page_fails_instead_of_becoming_empty(self):
        self.write_inputs([])
        with self.assertRaisesRegex(ValueError, "Missing inference pages"):
            self.build()

    def test_failed_page_fails_instead_of_becoming_empty(self):
        self.write_inputs([self.result(status="failed")])
        with self.assertRaisesRegex(ValueError, "did not succeed"):
            self.build()

    def test_duplicate_prediction_page_fails(self):
        self.write_inputs([self.result(), self.result()])
        with self.assertRaisesRegex(ValueError, "Duplicate prediction page"):
            self.build()

    def test_duplicate_physical_page_fails(self):
        other = dict(self.page, page_id="test/2", source_image_id=2, file_name="page2.png")
        self.write_inputs([self.result()], pages=[self.page, other])
        with self.assertRaisesRegex(ValueError, "Duplicate selected page or physical identity"):
            self.build()

    def test_prediction_dimension_mismatch_fails(self):
        self.write_inputs([self.result(width=1024)])
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            self.build()

    def test_candidate_accounting_mismatch_fails(self):
        self.write_inputs([self.result([self.prediction()], threshold_excluded=100)])
        with self.assertRaisesRegex(ValueError, "candidate accounting mismatch"):
            self.build()

    def test_prediction_protocol_hash_mismatch_fails(self):
        self.write_inputs([self.result(protocol_sha256="0" * 64)])
        with self.assertRaisesRegex(ValueError, "Prediction protocol hash mismatch"):
            self.build()

    def test_protocol_page_manifest_hash_mismatch_fails(self):
        self.write_inputs([self.result()])
        self.protocol["page_manifest_sha256"] = "0" * 64
        self.resign_protocol()
        with self.assertRaisesRegex(ValueError, "Protocol page manifest hash mismatch"):
            self.build()

    def test_protocol_category_mapping_mismatch_fails(self):
        self.write_inputs([self.result()])
        self.protocol["id2label"]["11"] = "Text"
        self.resign_protocol()
        with self.assertRaisesRegex(ValueError, "category map differs"):
            self.build()

    def test_modified_fixed_postprocessing_fails_even_with_matching_hashes(self):
        self.write_inputs([self.result()])
        self.protocol["postprocessing"]["clipping"] = True
        self.resign_protocol()
        with self.assertRaisesRegex(ValueError, "postprocessing differs"):
            self.build()

    def test_query_index_outside_fixed_model_range_fails(self):
        self.write_inputs([self.result([self.prediction(query=200)])])
        with self.assertRaisesRegex(ValueError, "Query index exceeds"):
            self.build()

    def test_score_above_one_fails(self):
        prediction = self.prediction()
        prediction["score"] = Decimal("1.00000001")
        self.write_inputs([self.result([prediction])])
        with self.assertRaisesRegex(ValueError, "Exported score violates"):
            self.build()

    def test_cross_class_and_duplicate_geometry_are_retained_with_shared_translation(self):
        # Sparse original top-k ranks are legal. Repeated query/class geometry
        # remains separate records; the GT category does not gate predictions.
        predictions = [self.prediction(rank=3, query=7), self.prediction(rank=91, query=7)]
        self.write_inputs([self.result(predictions)], [self.annotation(category=1)])
        report = self.build()
        self.assertEqual((report["r1_count"], report["r2_count"]), (1, 2))
        path = self.out / "doclaynet"
        r1, r2 = np.load(path / "R1.npy"), np.load(path / "R2.npy")
        self.assertTrue(np.array_equal(r2[0, 2:6], r2[1, 2:6]))
        group = pq.read_table(path / "groups.parquet").to_pylist()[0]
        shift = np.array([group["dx"], group["dy"], group["dx"], group["dy"]])
        self.assertTrue(np.array_equal(r1[0, 2:6] - shift, [-2000, -3000, 2000, 3000]))
        self.assertTrue(np.array_equal(r2[0, 2:6] - shift, [-1000, -2000, 2000, 2000]))
        metadata = pq.read_table(path / "R2_metadata.parquet").to_pylist()
        self.assertEqual([row["source_category_id"] for row in metadata], ["11", "11"])
        self.assertEqual([json.loads(row["attributes"])["prediction_index"] for row in metadata], [3, 91])
        self.assertEqual(report["model_label_to_doclaynet_name"]["0"], "N/A")
        self.assertEqual(report["model_label_to_doclaynet_name"]["11"], "Title")

    def test_fractional_xyxy_is_reconstructed_exactly_before_quantization(self):
        # More than the default Decimal context's 28 digits: rounding the
        # width before x0+width would move the right endpoint across its tie.
        box = tuple(map(Decimal, (
            "0.123456789012345678901234567890",
            "-0.000500000000000000000000000001",
            "0.123500000000000000000000000001",
            "0.001500000000000000000000000001",
        )))
        self.write_inputs([self.result([self.prediction(box=box)])], [self.annotation()])
        self.build()
        row = pq.read_table(self.out / "doclaynet" / "R2_metadata.parquet").to_pylist()[0]
        self.assertEqual(tuple(row[key] for key in ("local_x0", "local_y0", "local_x1", "local_y1")),
                         (123, -1, 124, 2))
        source_bbox = [Fraction(Decimal(value)) for value in json.loads(row["source_bbox"])]
        self.assertEqual(source_bbox[0] + source_bbox[2], Fraction(box[2]))
        self.assertEqual(source_bbox[1] + source_bbox[3], Fraction(box[3]))
        self.assertEqual(json.loads(row["attributes"])["source_xyxy"], list(map(str, box)))

    def test_unselected_gt_pages_are_accounted_without_failure(self):
        extra_image = {
            "id": 2, "file_name": "unselected.png", "width": 1025, "height": 1025,
            "doc_category": "scientific_articles", "collection": "collection",
            "doc_name": "document", "page_no": 2,
        }
        self.write_inputs([self.result()],
                          [self.annotation(), self.annotation(image_id=2, annotation_id=2)],
                          extra_images=[extra_image])
        report = self.build()
        self.assertEqual(report["filter_statistics"]["annotations.raw"], 2)
        self.assertEqual(report["filter_statistics"]["annotations.outside_selected_pages"], 1)
        self.assertEqual(report["filter_statistics"]["r1.raw"], 1)


if __name__ == "__main__":
    unittest.main()
