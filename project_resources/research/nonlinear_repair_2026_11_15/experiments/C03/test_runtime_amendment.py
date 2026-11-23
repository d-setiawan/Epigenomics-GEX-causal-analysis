"""Deterministic trust-boundary tests; no fit or evaluation data are read."""
import copy
import unittest

from runtime_amendment import admitted_protocol


class RuntimeAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "inputs": {"train.npz": "original-input"},
            "source_sha256": {"evaluate.py": "original-source"},
            "threadpools": [{"filepath": "numpy.so", "num_threads": 1}],
        }
        self.pools = self.expected["threadpools"] + [
            {"filepath": "mkl.so", "num_threads": 1},
            {"filepath": "omp.so", "num_threads": 1},
        ]
        self.actual = copy.deepcopy(self.expected)
        self.actual["threadpools"] = copy.deepcopy(self.pools)

    def test_exact_import_additions_only_and_no_input_mutation(self):
        before = copy.deepcopy(self.actual)
        self.assertEqual(admitted_protocol(self.expected, self.actual, self.pools), self.expected)
        self.assertEqual(self.actual, before)

    def test_changed_source_or_input_is_rejected(self):
        for field in ("inputs", "source_sha256"):
            with self.subTest(field=field):
                actual = copy.deepcopy(self.actual)
                actual[field] = {"changed": "hash"}
                with self.assertRaises(ValueError):
                    admitted_protocol(self.expected, actual, self.pools)

    def test_changed_library_threads_or_membership_is_rejected(self):
        variants = []
        actual = copy.deepcopy(self.actual)
        actual["threadpools"][1]["num_threads"] = 2
        variants.append(actual)
        actual = copy.deepcopy(self.actual)
        actual["threadpools"][1]["filepath"] = "unexpected.so"
        variants.append(actual)
        actual = copy.deepcopy(self.actual)
        actual["threadpools"].pop(0)
        variants.append(actual)
        actual = copy.deepcopy(self.actual)
        actual["threadpools"].append({"filepath": "extra.so", "num_threads": 1})
        variants.append(actual)
        for actual in variants:
            with self.subTest(actual=actual):
                with self.assertRaises(ValueError):
                    admitted_protocol(self.expected, actual, self.pools)


if __name__ == "__main__":
    unittest.main()
