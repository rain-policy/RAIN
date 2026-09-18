import json
from pathlib import Path
import unittest
from unittest.mock import patch
import run
import train


class ReleaseContractTests(unittest.TestCase):
    def test_manifest_has_eight_complete_models(self):
        tasks = run.load_tasks()
        self.assertEqual(len(tasks), 8)
        for task in tasks.values():
            self.assertEqual(set(task["files"]), {"uact", "flow", "heads", "route"})
            self.assertEqual(task["episodes"], 100)
            for role, filename in task["files"].items():
                self.assertTrue(
                    (run.ROOT / filename).resolve().is_relative_to(run.ROOT)
                )
                self.assertEqual(len(task["hashes"][role]), 64)

    def test_data_splits_are_disjoint(self):
        data = json.loads((run.ROOT / "manifests/datasets.json").read_text())
        for task in data.values():
            splits = [
                set(task["splits"][name]) for name in ["train", "calibration", "test"]
            ]
            self.assertEqual(list(map(len, splits)), [400, 50, 50])
            self.assertEqual(len(set.union(*splits)), 500)

    def test_checksum_failure_is_not_ignored(self):
        task = {"files": {"uact": "README.md"}, "hashes": {"uact": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "checksum"):
            run.verify(task)

    def test_training_preview_does_not_execute(self):
        with (
            patch(
                "sys.argv",
                [
                    "train.py",
                    "--task",
                    "pickcube",
                    "--demo",
                    "data/example.h5",
                    "--output",
                    "runs/example",
                ],
            ),
            patch("subprocess.run") as process,
            patch("builtins.print"),
        ):
            train.main()
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
