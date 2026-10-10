"""Download/load official YOLO11n and test inference without training."""
import argparse
from pathlib import Path
from training_workflow import sha256, write_json, runtime_info


def prepare(root):
    import numpy as np
    from ultralytics import YOLO
    folder = Path(root) / "pretrained"
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "yolo11n.pt"
    model = YOLO(str(weights))
    model.predict(np.zeros((640,640,3), dtype=np.uint8), device="cpu", save=False, verbose=False)
    result = {"weights": str(weights.resolve()), "sha256": sha256(weights),
              "source_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
              "load_and_cpu_predict": "passed", "trained": False, "environment": runtime_info()}
    write_json(folder / "yolo11n_manifest.json", result)
    print(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root)
