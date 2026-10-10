"""Single-GPU team training with verified, immutable Drive checkpoints.

Importing this module never starts training. Only train(..., approved=True) does.
Each training/resume attempt owns a new directory; no shared mutable leaderboard.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PINNED_ULTRALYTICS = "8.4.143"
WORKFLOW_VERSION = "2026-09-30-validator-fix-1"
PROJECT = Path(__file__).resolve().parents[1]
META_FILES = ("dataset_summary.json", "dataset_manifest_v001.csv", "split_v001.csv", "fire_v001_colab.yaml", "label_corrections.json")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    expected = sha256(source)
    shutil.copyfile(source, tmp)
    if sha256(tmp) != expected:
        raise IOError(f"Copy checksum mismatch: {destination}")
    os.replace(tmp, destination)
    return expected


def make_id(member):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", member):
        raise ValueError("MEMBER_ID: use 1-32 English letters, digits, _ or -")
    return f"{member}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"


def runtime_info():
    import torch
    import ultralytics
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def verify_prepared_dataset(dataset):
    """Check manifests AND current labels; unchanged metadata alone is insufficient."""
    from prepare_dfire import LABEL_POLICY
    import yaml
    dataset = Path(dataset).resolve()
    for name in META_FILES:
        if not (dataset / name).is_file():
            raise FileNotFoundError(dataset / name)
    summary = read_json(dataset / "dataset_summary.json")
    if summary.get("label_policy") != LABEL_POLICY:
        raise ValueError("Prepare data with the current label policy")
    for filename, key in (("dataset_manifest_v001.csv", "manifest_sha256"),
                          ("label_corrections.json", "corrections_sha256")):
        if sha256(dataset / filename) != summary.get(key):
            raise ValueError(f"Prepared metadata changed: {filename}")
    config = yaml.safe_load((dataset / "fire_v001_colab.yaml").read_text(encoding="utf-8"))
    if Path(config["path"]).resolve() != dataset or config.get("names") != {0: "smoke", 1: "fire"}:
        raise ValueError("Data YAML path/classes differ from prepared dataset")
    for split in ("train", "val", "test"):
        if config.get(split) != f"{split}/images":
            raise ValueError(f"Unexpected YAML {split} path")
    with (dataset / "dataset_manifest_v001.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    expected_rows = []
    names = set()
    for row in manifest:
        split, name = row["split"], row["file_name"]
        if split not in ("train", "val", "test") or name in names or Path(name).name != name:
            raise ValueError("Duplicate/invalid image in dataset manifest")
        names.add(name)
        for key in ("image_path", "label_path"):
            target = (dataset / row[key]).resolve()
            expected_name = name if key == "image_path" else f"{Path(name).stem}.txt"
            folder = "images" if key == "image_path" else "labels"
            if row[key] != f"{split}/{folder}/{expected_name}" or not target.is_file():
                raise ValueError(f"Invalid/missing dataset file: {row[key]}")
        if sha256(dataset / row["label_path"]) != row["label_sha256"]:
            raise ValueError(f"Prepared label changed: {row['label_path']}")
        expected_rows.append({k: row[k] for k in ("file_name", "split", "split_source")})
    with (dataset / "split_v001.csv").open(encoding="utf-8-sig", newline="") as handle:
        if list(csv.DictReader(handle)) != expected_rows:
            raise ValueError("Split CSV differs from dataset manifest")
    for split in ("train", "val", "test"):
        rows = [r for r in manifest if r["split"] == split]
        if len(rows) != summary["splits"][split]["images"]:
            raise ValueError(f"Manifest count mismatch: {split}")
        from prepare_dfire import image_names
        if image_names(dataset / split / "images") != {r["file_name"] for r in rows}:
            raise ValueError(f"Image directory differs from manifest: {split}")
        if {p.name for p in (dataset / split / "labels").glob('*.txt')} != {Path(r['label_path']).name for r in rows}:
            raise ValueError(f"Label directory differs from manifest: {split}")
    return summary


def preflight(dataset, drive_results, require_gpu=True):
    """Check prepared data, free space, pinned library and Drive write/read access."""
    dataset, drive_results = Path(dataset), Path(drive_results)
    info = runtime_info()
    if info["ultralytics"] != PINNED_ULTRALYTICS:
        raise RuntimeError(f"Install requirements-colab.txt: {info['ultralytics']}")
    if require_gpu and not info["gpu"]:
        raise RuntimeError("Select a Colab GPU runtime before training")
    summary = verify_prepared_dataset(dataset)
    if summary.get("classes") != {"0": "smoke", "1": "fire"} or summary.get("fold") != 1:
        raise ValueError("Expected D-Fire fold 1 with 0=smoke, 1=fire")
    if summary.get("source_sha256") is None or not summary.get("images_verified"):
        raise ValueError("Prepare data with the current script and --verify-images")
    expected = {"train": 13776, "val": 3445, "test": 4306}
    for split, count in expected.items():
        if summary["splits"][split]["images"] != count:
            raise ValueError(f"Unexpected {split} count")
    if shutil.disk_usage(dataset).free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free local storage is required after extraction")
    probe = drive_results / "preflight" / (make_id("writecheck") + ".json")
    write_json(probe, {"write_read_test": True, "environment": info})
    if not read_json(probe)["write_read_test"]:
        raise IOError("Drive read-back failed")
    return {"environment": info, "splits": summary["splits"], "drive_probe": str(probe)}


def save_checkpoint(trainer, run_dir):
    """Called after optimizer-inclusive last.pt is saved, before final stripping.

    Incomplete copies cannot advance latest_checkpoint.json. Keeping each epoch
    also leaves a fallback if Drive synchronization is interrupted.
    """
    run_dir = Path(run_dir)
    epoch = int(trainer.epoch) + 1
    target = run_dir / "checkpoints" / f"epoch_{epoch:04d}"
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    sources = {"last.pt": Path(trainer.last), "best.pt": Path(trainer.best),
               "results.csv": Path(trainer.save_dir) / "results.csv"}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint file missing: {source}")
        hashes[name] = copy_verified(source, target / name)
    manifest = {"epoch": epoch, "files": hashes}
    write_json(target / "manifest.json", manifest)
    write_json(run_dir / "latest_checkpoint.json", {"directory": str(target.relative_to(run_dir)), **manifest})
    write_json(run_dir / "status.json", {"state": "training", "epoch": epoch, "backup_verified": True})
    print(f"[Drive backup verified] epoch={epoch}: {target}", flush=True)


def restore_checkpoint(parent, local_run, summary):
    parent, local_run = Path(parent), Path(local_run)
    previous = read_json(parent / "dataset_summary.json")
    for key in ("source_sha256", "fold", "classes", "label_policy", "manifest_sha256", "corrections_sha256"):
        if previous.get(key) != summary.get(key):
            raise ValueError(f"Resume dataset mismatch: {key}")
    if sha256(parent / "split_v001.csv") != sha256(Path(summary["yaml"]).parent / "split_v001.csv"):
        raise ValueError("Resume split manifest mismatch")
    pointer = read_json(parent / "latest_checkpoint.json")
    folder = (parent / pointer["directory"]).resolve()
    if not folder.is_relative_to(parent.resolve()):
        raise ValueError("Invalid checkpoint directory")
    if read_json(folder / "manifest.json") != {"epoch": pointer["epoch"], "files": pointer["files"]}:
        raise ValueError("Incomplete checkpoint manifest")
    for name in ("last.pt", "best.pt", "results.csv"):
        if sha256(folder / name) != pointer["files"][name]:
            raise ValueError(f"Corrupt checkpoint: {name}")
    local_run.mkdir(parents=True, exist_ok=False)
    for name in ("last.pt", "best.pt"):
        copy_verified(folder / name, local_run / "weights" / name)
    copy_verified(folder / "results.csv", local_run / "results.csv")
    return local_run / "weights" / "last.pt"


def train(dataset, drive_results, local_runs, pretrained, member, *, approved=False,
          labels_reviewed=False, batch=16, resume_from=None):
    if not approved or not labels_reviewed:
        raise ValueError("Training is disabled until RUN_FULL_TRAIN and LABELS_REVIEWED are True")
    preflight(dataset, drive_results)
    from ultralytics import YOLO, settings
    # Avoid MLflow/W&B auto-integration inherited from another machine.
    settings.update({"mlflow": False, "wandb": False})
    dataset = Path(dataset).resolve()
    config = read_json(PROJECT / "configs" / "train_yolo11n_v1.json")
    if batch not in (8, 16):
        raise ValueError("Use batch=16, or batch=8 for a new run after an OOM")
    config["batch"] = batch
    name = make_id(member)
    run_dir = Path(drive_results) / "yolo11n_v1" / name
    local_run = (Path(local_runs) / name).resolve()
    if local_run.exists():
        raise FileExistsError(local_run)
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = read_json(dataset / "dataset_summary.json")
    write_json(run_dir / "status.json", {"state": "initializing"})
    for filename in META_FILES:
        copy_verified(dataset / filename, run_dir / filename)
    # Snapshot all executable inputs so later shared-folder edits do not erase provenance.
    for relative in ("scripts/training_workflow.py", "scripts/prepare_dfire.py",
                     "configs/train_yolo11n_v1.json", "requirements-colab.txt",
                     "notebooks/dfire_team_training.ipynb"):
        copy_verified(PROJECT / relative, run_dir / "source" / relative)
    write_json(run_dir / "environment.json", runtime_info())
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                              capture_output=True, text=True).stdout
    (run_dir / "pip-freeze.txt").write_text(packages, encoding="utf-8")
    record = {"member": member, "run_id": name, "model": "yolo11n.pt", "config": config,
              "parent_run": str(resume_from) if resume_from else None,
              "split_sha256": sha256(dataset / "split_v001.csv"),
              "source_sha256": summary["source_sha256"]}
    record["dataset_manifest_sha256"] = summary["manifest_sha256"]
    if resume_from:
        parent = Path(resume_from)
        previous = read_json(parent / "run.json")
        if previous["config"] != config:
            raise ValueError("Resume must use the same configuration (including batch)")
        if read_json(parent / "status.json")["state"] in ("trained", "validated"):
            raise ValueError("This run already completed; evaluate its final/best.pt")
        checkpoint = restore_checkpoint(parent, local_run, summary)
        model = YOLO(str(checkpoint))
        ckpt = model.ckpt
        if not ckpt or ckpt.get("epoch", -1) < 0 or ckpt.get("optimizer") is None:
            raise ValueError("Checkpoint is not resumable; refusing to silently start a new training")
        if int(ckpt["epoch"]) + 1 >= config["epochs"]:
            raise ValueError("Checkpoint already reached the target epochs; recover best.pt for evaluation instead")
        old_data = ckpt.get("train_args", {}).get("data")
        if (isinstance(old_data, str) and Path(old_data).exists()
                and Path(old_data).resolve() != dataset / "fire_v001_colab.yaml"):
            raise ValueError("Resume would read an old existing YAML; use the original LOCAL_DATASET path")
        args = dict(resume=True, data=str(dataset / "fire_v001_colab.yaml"),
                    save_dir=str(local_run), device=0, workers=2, batch=batch)
        record["pretrained_sha256"] = previous["pretrained_sha256"]
    else:
        pretrained = Path(pretrained)
        if pretrained.name != "yolo11n.pt" or not pretrained.is_file():
            raise ValueError("Provide the preflight-verified yolo11n.pt file")
        record["pretrained_sha256"] = sha256(pretrained)
        model = YOLO(str(pretrained))
        args = {k: v for k, v in config.items() if k != "model"}
        args.update(data=str(dataset / "fire_v001_colab.yaml"), project=str(local_run.parent),
                    name=name, save_dir=str(local_run), exist_ok=False)
    write_json(run_dir / "run.json", record)
    model.add_callback("on_model_save", lambda trainer: save_checkpoint(trainer, run_dir))
    print(f"RUN_DIR = {run_dir}", flush=True)
    try:
        model.train(**args)
        actual = Path(model.trainer.save_dir)
        final = run_dir / "final"
        final.mkdir(exist_ok=False)
        for source in actual.rglob("*"):
            if source.is_file():
                copy_verified(source, final / source.relative_to(actual))
        copy_verified(actual / "weights" / "best.pt", final / "best.pt")
        write_json(run_dir / "status.json", {"state": "trained", "model": "final/best.pt"})
        print(f"Training saved: {run_dir}", flush=True)
        return run_dir
    except BaseException as exc:
        try:
            write_json(run_dir / "status.json", {"state": "interrupted", "error": str(exc),
                       "resume_from": str(run_dir), "has_checkpoint": (run_dir / "latest_checkpoint.json").exists()})
        except OSError as status_error:
            print(f"Could not update Drive status: {status_error}. Original error: {exc}", file=sys.stderr)
        raise


def evaluate(run_dir, dataset, local_runs, split="val", *, test_approved=False, device=0):
    if split not in ("val", "test"):
        raise ValueError(split)
    if split == "test" and not test_approved:
        raise ValueError("Choose ONE final run using validation before enabling test")
    from ultralytics import YOLO
    run_dir, dataset = Path(run_dir), Path(dataset)
    summary = verify_prepared_dataset(dataset)
    saved = read_json(run_dir / "run.json")
    if saved["split_sha256"] != sha256(dataset / "split_v001.csv"):
        raise ValueError("Evaluation dataset split differs from training")
    if saved["source_sha256"] != summary["source_sha256"]:
        raise ValueError("Evaluation archive hashes differ from training")
    if saved["dataset_manifest_sha256"] != sha256(dataset / "dataset_manifest_v001.csv"):
        raise ValueError("Evaluation labels differ from training")
    model_path = run_dir / "final" / "best.pt"
    name = make_id(split)
    model = YOLO(str(model_path))
    if model.names != {0: "smoke", 1: "fire"}:
        raise ValueError("Expected a trained D-Fire model with 0=smoke, 1=fire")
    validation_output = {}

    def remember_output(validator):
        validation_output["save_dir"] = Path(validator.save_dir).resolve()

    model.add_callback("on_val_end", remember_output)
    metrics = model.val(data=str(dataset / "fire_v001_colab.yaml"), split=split,
                        imgsz=640, batch=saved["config"]["batch"], device=device, plots=True,
                        project=str(Path(local_runs).resolve()), name=name)
    if "save_dir" not in validation_output or not validation_output["save_dir"].is_dir():
        raise RuntimeError("Validation completed without a usable output directory")
    actual_output = validation_output["save_dir"]
    directory = run_dir / "evaluations" / name
    directory.mkdir(parents=True, exist_ok=False)
    for source in actual_output.rglob("*"):
        if source.is_file():
            copy_verified(source, directory / source.relative_to(actual_output))
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(index)
        per_class.append(dict(class_id=int(class_id), name=model.names[int(class_id)],
                              precision=float(p), recall=float(r), mAP50=float(ap50), mAP50_95=float(ap)))
    result = {"split": split, "model_sha256": sha256(model_path),
              "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
              "per_class": per_class, "speed_ms": metrics.speed}
    write_json(directory / "metrics.json", result)
    if split == "val":
        write_json(run_dir / "validation_summary.json", result)
        write_json(run_dir / "status.json", {"state": "validated", "model": "final/best.pt"})
    return result


def collect_results(drive_results):
    """Read-only comparison; returns rows, does not select or merge models."""
    rows = []
    for path in sorted((Path(drive_results) / "yolo11n_v1").glob("*/run.json")):
        run = read_json(path)
        status = read_json(path.parent / "status.json")
        row = {"run_id": run["run_id"], "member": run["member"], "state": status["state"],
               "batch": run["config"]["batch"], "seed": run["config"]["seed"],
               "split_sha256": run["split_sha256"], "pretrained_sha256": run["pretrained_sha256"],
               "dataset_manifest_sha256": run["dataset_manifest_sha256"]}
        result = path.parent / "validation_summary.json"
        if result.exists():
            row.update(read_json(result)["metrics"])
        rows.append(row)
    return rows
