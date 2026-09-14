"""Regression checks for slab certificates and strict half-open overlaps."""

import ctypes
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from validate import load_counter, validate
from vosma_dataset.common import Dataset


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ValidationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler_directory = tempfile.TemporaryDirectory(
            prefix="vosma-validation-counter-"
        )
        cls.addClassCleanup(cls.compiler_directory.cleanup)
        library = pathlib.Path(cls.compiler_directory.name) / "overlap_count.so"
        subprocess.run(
            [
                "c++", "-O2", "-std=c++17", "-shared", "-fPIC",
                str(ROOT / "overlap_count.cpp"), "-o", str(library),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cls.counter = staticmethod(load_counter(library))

    @staticmethod
    def records(boxes):
        result = np.zeros((len(boxes), 7), dtype="<i8")
        result[:, 0] = np.arange(len(boxes))
        if boxes:
            result[:, 2:6] = boxes
        result[:, 6] = 1
        return result

    def count(self, left, right):
        left, right = self.records(left), self.records(right)
        result = ctypes.c_uint64()
        code = self.counter(
            left.ctypes.data,
            right.ctypes.data,
            len(left),
            len(right),
            ctypes.byref(result),
        )
        self.assertEqual(code, 0)
        return result.value

    def test_half_open_boundaries_duplicates_and_simultaneous_starts(self):
        cases = [
            ("x edge only", [(0, 0, 2, 2)], [(2, 0, 4, 2)], 0),
            ("y edge only", [(0, 0, 2, 2)], [(0, 2, 2, 4)], 0),
            ("corner only", [(0, 0, 2, 2)], [(2, 2, 4, 4)], 0),
            ("positive overlap", [(0, 0, 2, 2)], [(1, 1, 3, 3)], 1),
            ("contained", [(0, 0, 4, 4)], [(1, 1, 2, 2)], 1),
            ("duplicate occurrences", [(0, 0, 2, 2)] * 3,
             [(0, 0, 2, 2)] * 2, 6),
            ("same x starts", [(0, 0, 3, 3), (0, 5, 3, 8)],
             [(0, 1, 1, 2), (0, 6, 1, 7), (0, 9, 1, 10)], 2),
            ("end and start at same x", [(0, 0, 2, 2), (2, 0, 4, 2)],
             [(2, 0, 3, 2)], 1),
            ("negative local coordinates", [(-3, -3, -1, -1)],
             [(-2, -2, 0, 0)], 1),
            ("empty side", [], [(0, 0, 2, 2)], 0),
        ]
        for name, left, right, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(self.count(left, right), expected)
                self.assertEqual(self.count(right, left), expected)

    @staticmethod
    def build_toy(directory):
        dataset = Dataset("toy", directory)
        for index in range(2):
            key = dataset.group(
                ["frame", str(index)],
                image_key=str(index),
                split="test",
                width=10,
                height=10,
            )
            for side in ("r1", "r2"):
                dataset.stats[f"{side}.raw"] += 1
                dataset.add(
                    key, side, [0, 0, 2, 2], f"{side}.txt", index + 1,
                    annotation_id=str(index), image_id=str(index),
                )
        dataset.finish()
        return pathlib.Path(directory) / "toy"

    def test_valid_toy_then_reject_inaccurate_declared_slab(self):
        with tempfile.TemporaryDirectory(prefix="vosma-validation-toy-") as tmp:
            path = self.build_toy(tmp)
            report = validate(path, self.counter)
            self.assertTrue(report["passed"])
            self.assertEqual(report["within_group_overlap_count"], 2)
            self.assertEqual(report["cross_group_intersections"], 0)

            # Alter only the declared extent, leaving every rectangle, local
            # metadata coordinate and translation unchanged. The declared
            # slabs remain disjoint, but the first slab no longer contains
            # the actual rectangles. A certificate-only check misses this.
            groups_path = path / "groups.parquet"
            table = pq.read_table(groups_path)
            maxima = table["global_max_x"].to_pylist()
            minima = table["global_min_x"].to_pylist()
            self.assertGreater(maxima[0], minima[0] + 1)
            maxima[0] = minima[0] + 1
            self.assertLess(maxima[0], minima[1])
            field_index = table.schema.get_field_index("global_max_x")
            table = table.set_column(
                field_index, table.schema.field(field_index),
                pa.array(maxima, type=pa.int64()),
            )
            pq.write_table(table, groups_path, compression="zstd")

            with self.assertRaises(AssertionError):
                validate(path, self.counter)


if __name__ == "__main__":
    unittest.main()
