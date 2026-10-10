"""Read archives without extraction; optional full image decode and YOLO11 preflight.
Never trains. All outputs must be inside the explicitly selected dataset directory.
"""
import argparse
import io
import zipfile
from collections import Counter
from pathlib import Path

from training_workflow import copy_verified, make_id, sha256, write_json, runtime_info
from prepare_dfire import read_split, require_equal, normalize_label_text, LABEL_POLICY


def audit(root, verify_images=False):
    root = Path(root)
    dataset, splits = root / "D-Fire.zip", root / "d-fire 텍스트 분할.zip"
    print("Computing archive hashes...", flush=True)
    hashes = {"dataset": sha256(dataset), "splits": sha256(splits)}
    with zipfile.ZipFile(splits) as z:
        prefix = "Data splitting/5-fold cross validation"
        groups = {"train": set(read_split(z, prefix + "/dfire_train1.txt")),
                  "val": set(read_split(z, prefix + "/dfire_valid1.txt")),
                  "test": set(read_split(z, "Data splitting/dfire_test.txt"))}
    if groups["train"] & groups["val"] or groups["test"] & (groups["train"] | groups["val"]):
        raise ValueError("Overlapping splits")
    boxes, negatives, corrections = Counter(), 0, []
    with zipfile.ZipFile(dataset) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        images = [n for n in names if Path(n).suffix.lower() in (".jpg", ".jpeg", ".png")]
        for part, wanted in (("train", groups["train"] | groups["val"]), ("test", groups["test"])):
            actual = {Path(n).name for n in images if n.startswith(part + "/images/")}
            require_equal(part, wanted, actual)
        expected_labels = {n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt" for n in images}
        actual_labels = {n for n in names if "/labels/" in n and n.endswith(".txt")}
        require_equal("archive image-label pairs", expected_labels, actual_labels)
        for index, image in enumerate(images, 1):
            label = image.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            original = z.read(label).decode("utf-8-sig")
            cleaned, changes = normalize_label_text(original, label)
            corrections.extend(changes)
            lines = cleaned.strip().splitlines()
            negatives += not lines
            for line in lines:
                parts = line.split()
                if len(parts) != 5 or parts[0] not in ("0", "1"):
                    raise ValueError(f"Invalid label {label}")
                coords = list(map(float, parts[1:]))
                if not all(0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                    raise ValueError(f"Invalid coordinates {label}")
                boxes[int(parts[0])] += 1
            if verify_images:
                from PIL import Image
                with Image.open(io.BytesIO(z.read(image))) as opened:
                    opened.load()
            if index % 2000 == 0:
                print(f"Checked {index}/{len(images)} image-label pairs", flush=True)
    return {"source_sha256": hashes, "fold": 1, "classes": {"0": "smoke", "1": "fire"},
            "split_counts": {k: len(v) for k, v in groups.items()}, "total_images": len(images),
            "negative_images": negatives, "boxes": dict(boxes), "images_decoded": verify_images,
            "label_policy": LABEL_POLICY, "correction_counts": dict(Counter(c['action'] for c in corrections)),
            "corrections": corrections}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--model-check", action="store_true")
    args = parser.parse_args()
    report = audit(args.root, args.verify_images)
    report["environment"] = runtime_info()
    dest = args.root / "preflight_reports" / (make_id("local") + ".json")
    write_json(dest, report)
    print(f"Archive audit saved: {dest}", flush=True)
    if args.model_check:
        from prepare_pretrained import prepare
        report["model_check"] = prepare(args.root)
    write_json(dest, report)
    print(f"Saved: {dest}", flush=True)
    print({k: v for k, v in report.items() if k != 'corrections'}, flush=True)
