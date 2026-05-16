import unittest
from unittest import mock

from scripts import eval_libero100_headless as eval_headless


class _DummySuite:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks


class Libero100EvalHeadlessTest(unittest.TestCase):
    def test_default_task_suite_is_libero_100(self):
        args = eval_headless.parse_args(
            [
                "--checkpoint",
                "/tmp/checkpoint",
                "--output_dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.task_suite_name, "libero_100")

    def test_libero_100_plans_libero_90_then_libero_10(self):
        def fake_load(suite_name, category_value):
            del category_value
            return _DummySuite(90 if suite_name == "libero_90" else 10)

        with mock.patch.object(eval_headless, "_load_benchmark", side_effect=fake_load):
            plan = eval_headless._suite_plan("libero_100", "unused")

        self.assertEqual([suite for suite, _ in plan], ["libero_90", "libero_10"])
        self.assertEqual([suite.n_tasks for _, suite in plan], [90, 10])

    def test_multi_suite_task_ids_require_suite_prefixes(self):
        suites = [("libero_90", _DummySuite(90)), ("libero_10", _DummySuite(10))]
        selected = eval_headless._suite_task_ids("libero_90:0,2,libero_10:0-1", suites)

        self.assertEqual(selected, {"libero_90": [0, 2], "libero_10": [0, 1]})
        with self.assertRaisesRegex(ValueError, "must be prefixed"):
            eval_headless._suite_task_ids("0,1", suites)


if __name__ == "__main__":
    unittest.main()
