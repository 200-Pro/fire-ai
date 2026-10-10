"""Prepare the official D-Fire archives for Ultralytics YOLO.

The script extracts the dataset into local scratch storage, applies one of the
official five-fold validation splits, audits every image/label pair, and writes
reproducibility metadata.  It never deletes an existing directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {0: "smoke", 1: "fire"}
LABEL_POLICY = "v2_clip_to_image_drop_degenerate_deduplicate"


def normalize_label_text(text, source):
    """Deterministic geometry repair, identical for train/val/test; no relabeling."""
    clean, changes, seen = [], [], set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] not in ("0", "1"):
            raise ValueError(f"{source}:{number}: invalid class/columns")
        x, y, w, h = map(float, parts[1:])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w < 0 or h < 0:
            raise ValueError(f"{source}:{number}: non-finite or negative box")
        reason, output = None, line.strip()
        if w == 0 or h == 0:
            reason, output = "drop_zero_area", ""
        else:
            x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
            clipped = (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))
            a, b, c, d = clipped
            if c <= a or d <= b:
                reason, output = "drop_outside_image", ""
            elif max(abs(u-v) for u, v in zip((x1,y1,x2,y2), clipped)) > 1e-12:
                output = f"{parts[0]} {(a+c)/2:.12g} {(b+d)/2:.12g} {c-a:.12g} {d-b:.12g}"
                reason = "clip_to_image"
        if output:
            key = tuple(map(float, output.split()))
            if key in seen:
                reason, output = "drop_duplicate", ""
            else:
                seen.add(key)
        if reason:
            changes.append({"source": str(source), "line": number, "action": reason,
                            "original": line, "replacement": output})
        if output:
            clean.append(output)
    return "\n".join(clean) + ("\n" if clean else ""), changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--split-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        candidate = (destination / member.filename).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"Unsafe ZIP entry: {member.filename}")
    archive.extractall(destination)


def read_split(archive: zipfile.ZipFile, name: str) -> list[str]:
    with archive.open(name) as handle:
        text = handle.read().decode("utf-8-sig")
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate names in split: {name}")
    if any(Path(n).name != n or "\\" in n for n in names):
        raise ValueError(f"Expected filenames only: {name}")
    return names


def image_names(directory: Path) -> set[str]:
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def require_equal(label: str, left: set[str], right: set[str]) -> None:
    if left == right:
        return
    missing = sorted(left - right)[:10]
    extra = sorted(right - left)[:10]
    raise ValueError(f"{label} mismatch: missing={missing}, extra={extra}")


def validate_pair_names(root: Path, split: str) -> tuple[list[Path], list[Path]]:
    images = sorted(
        path
        for path in (root / split / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sorted((root / split / "labels").glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    require_equal(f"{split} image/label", image_stems, label_stems)
    return images, labels


def audit_label(path: Path) -> tuple[int, set[int]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return 0, set()

    classes: set[int] = set()
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric label") from exc
        if class_id not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number}: invalid class {class_id}")
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{path}:{line_number}: coordinates outside 0..1")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number}: non-positive box size")
        classes.add(class_id)
    return len(lines), classes


def prepare(args: argparse.Namespace) -> dict[str, object]:
    dataset_zip = args.dataset_zip.resolve()
    split_zip = args.split_zip.resolve()
    output = args.output.resolve()

    for archive_path in (dataset_zip, split_zip):
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    sources = {"dataset": digest(dataset_zip), "splits": digest(split_zip)}
    marker = output / "dataset_summary.json"
    reused = False
    if output.exists() and any(output.iterdir()):
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if (saved.get("source_sha256") != sources or saved.get("fold") != args.fold
                    or saved.get("label_policy") != LABEL_POLICY):
                raise ValueError("Existing data have different/unknown hashes or fold; use a new output directory")
            reused = True
        else:
            raise FileExistsError(
                f"Output is not empty: {output}. Choose a new /content directory."
            )

    output.mkdir(parents=True, exist_ok=True)
    if not reused:
        with zipfile.ZipFile(dataset_zip) as archive:
            safe_extract(archive, output)

    required = [
        output / "train" / "images",
        output / "train" / "labels",
        output / "test" / "images",
        output / "test" / "labels",
    ]
    for path in required:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing expected directory: {path}")

    with zipfile.ZipFile(split_zip) as archive:
        prefix = "Data splitting/5-fold cross validation"
        official_train = read_split(archive, f"{prefix}/dfire_train{args.fold}.txt")
        official_val = read_split(archive, f"{prefix}/dfire_valid{args.fold}.txt")
        official_test = read_split(archive, "Data splitting/dfire_test.txt")

    train_names = image_names(output / "train" / "images")
    test_names = image_names(output / "test" / "images")
    official_train_set = set(official_train)
    official_val_set = set(official_val)
    official_test_set = set(official_test)

    if (official_train_set & official_val_set or official_train_set & official_test_set
            or official_val_set & official_test_set):
        raise ValueError("Official train/validation/test lists overlap")
    require_equal(
        "official train+validation",
        train_names,
        official_train_set if reused else official_train_set | official_val_set,
    )
    require_equal("official test", test_names, official_test_set)

    if reused:
        require_equal("official validation", image_names(output / "val" / "images"), official_val_set)
    else:
        (output / "val" / "images").mkdir(parents=True)
        (output / "val" / "labels").mkdir(parents=True)
        for image_name in official_val:
            source_image = output / "train" / "images" / image_name
            source_label = output / "train" / "labels" / f"{Path(image_name).stem}.txt"
            if not source_image.is_file() or not source_label.is_file():
                raise FileNotFoundError(f"Missing validation pair for {image_name}")
            shutil.move(source_image, output / "val" / "images" / image_name)
            shutil.move(source_label, output / "val" / "labels" / source_label.name)

    corrections_path = output / "label_corrections.json"
    corrections = json.loads(corrections_path.read_text(encoding="utf-8")) if reused else []
    if reused:
        if digest(corrections_path) != saved.get("corrections_sha256"):
            raise ValueError("Saved correction log changed; prepare a fresh dataset")
        old_manifest = output / "dataset_manifest_v001.csv"
        if digest(old_manifest) != saved.get("manifest_sha256"):
            raise ValueError("Saved manifest changed; prepare a fresh dataset")
        with old_manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                label = (output / row["label_path"]).resolve()
                if not label.is_relative_to(output) or digest(label) != row["label_sha256"]:
                    raise ValueError("Prepared label changed; prepare a fresh dataset")
    if not reused:
        for split in ("train", "val", "test"):
            for label in sorted((output / split / "labels").glob("*.txt")):
                text = label.read_text(encoding="utf-8-sig")
                cleaned, changes = normalize_label_text(text, label.relative_to(output).as_posix())
                if changes:
                    label.write_text(cleaned, encoding="utf-8", newline="\n")
                    corrections.extend(changes)
        corrections_path.write_text(json.dumps(corrections, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    manifest_path = output / "dataset_manifest_v001.csv"
    split_path = output / "split_v001.csv"
    class_boxes: Counter[int] = Counter()
    split_summary: dict[str, dict[str, int]] = {}
    manifest_rows: list[dict[str, object]] = []

    for split in ("train", "val", "test"):
        images, _ = validate_pair_names(output, split)
        negatives = 0
        boxes = 0
        for image in images:
            if getattr(args, "verify_images", False):
                from PIL import Image
                with Image.open(image) as opened:
                    opened.load()
            label = output / split / "labels" / f"{image.stem}.txt"
            box_count, classes = audit_label(label)
            boxes += box_count
            if box_count == 0:
                negatives += 1
            if box_count:
                for line in label.read_text(encoding="utf-8-sig").splitlines():
                    class_boxes[int(line.split()[0])] += 1
            manifest_rows.append(
                {
                    "file_name": image.name,
                    "split": split,
                    "image_path": image.relative_to(output).as_posix(),
                    "label_path": label.relative_to(output).as_posix(),
                    "box_count": box_count,
                    "classes": "|".join(CLASS_NAMES[item] for item in sorted(classes)),
                    "is_negative": box_count == 0,
                    "label_sha256": digest(label),
                    "split_source": f"official_dfire_fold_{args.fold}",
                }
            )
        split_summary[split] = {
            "images": len(images),
            "labels": len(images),
            "boxes": boxes,
            "negative_images": negatives,
        }

    fields = list(manifest_rows[0])
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with split_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "split", "split_source"])
        for row in manifest_rows:
            writer.writerow([row["file_name"], row["split"], row["split_source"]])

    yaml_path = output / "fire_v001_colab.yaml"
    yaml_path.write_text(
        f"path: {output.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n",
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "dataset": "D-Fire",
        "fold": args.fold,
        "source_sha256": sources,
        "images_verified": bool(getattr(args, "verify_images", False)),
        "label_policy": LABEL_POLICY,
        "label_corrections": len(corrections),
        "manifest_sha256": digest(manifest_path),
        "corrections_sha256": digest(corrections_path),
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "class_box_counts": {str(key): class_boxes[key] for key in CLASS_NAMES},
        "splits": split_summary,
        "total_images": len(manifest_rows),
        "yaml": str(yaml_path),
    }
    marker.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    try:
        prepare(parse_args())
    except Exception as exc:  # concise CLI failure for Colab cells
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
