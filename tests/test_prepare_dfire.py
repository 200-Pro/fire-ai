import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_dfire.py"


class PrepareDFireTest(unittest.TestCase):
    def test_prepares_official_fold_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_zip = root / "dataset.zip"
            split_zip = root / "splits.zip"
            output = root / "prepared"

            train_names = [f"AoF{i:05d}.jpg" for i in range(5)]
            test_names = ["AoF99999.jpg"]
            with zipfile.ZipFile(dataset_zip, "w") as archive:
                for name in train_names:
                    archive.writestr(f"train/images/{name}", b"fake-image")
                    archive.writestr(
                        f"train/labels/{Path(name).stem}.txt",
                        "0 0.5 0.5 0.2 0.2\n" if name != train_names[-1] else "",
                    )
                for name in test_names:
                    archive.writestr(f"test/images/{name}", b"fake-image")
                    archive.writestr(
                        f"test/labels/{Path(name).stem}.txt",
                        "1 0.5 0.5 0.3 0.3\n",
                    )

            with zipfile.ZipFile(split_zip, "w") as archive:
                prefix = "Data splitting/5-fold cross validation"
                for fold in range(1, 6):
                    valid = [train_names[fold - 1]]
                    train = [name for name in train_names if name not in valid]
                    archive.writestr(
                        f"{prefix}/dfire_train{fold}.txt", "\n".join(train)
                    )
                    archive.writestr(
                        f"{prefix}/dfire_valid{fold}.txt", "\n".join(valid)
                    )
                archive.writestr(
                    "Data splitting/dfire_test.txt", "\n".join(test_names)
                )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-zip",
                    str(dataset_zip),
                    "--split-zip",
                    str(split_zip),
                    "--output",
                    str(output),
                    "--fold",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0)
            summary = json.loads(
                (output / "dataset_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["splits"]["train"]["images"], 4)
            self.assertEqual(summary["splits"]["val"]["images"], 1)
            self.assertEqual(summary["splits"]["test"]["images"], 1)
            self.assertEqual(summary["class_box_counts"], {"0": 4, "1": 1})
            self.assertTrue((output / "fire_v001_colab.yaml").is_file())
            self.assertTrue((output / "label_corrections.json").is_file())
            # Repeat revalidates rather than trusting a stale completion marker.
            subprocess.run(completed.args, check=True, capture_output=True)
            changed_fold = list(completed.args)
            changed_fold[-1] = "2"
            bad = subprocess.run(changed_fold, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            (output / 'train' / 'labels' / 'AoF00001.txt').write_text('0 0.5 0.5 0 0', encoding='utf-8')
            bad = subprocess.run(completed.args, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
