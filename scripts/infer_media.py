"""Load the team's trained best.pt and stream per-frame detections to JSONL."""
import argparse
import json
from pathlib import Path

from training_workflow import make_id, sha256, write_json


def predict(weights, source, output, confidence=0.25, device="cpu", save_video=False):
    from ultralytics import YOLO
    weights, source = Path(weights), Path(source)
    if not weights.is_file() or not source.is_file():
        raise FileNotFoundError("Provide an existing best.pt and local image/MP4")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    model = YOLO(str(weights))
    if model.names != {0: "smoke", 1: "fire"}:
        raise ValueError("Expected a trained D-Fire model, not the COCO pretrained model")
    name = make_id("predict")
    target = Path(output).resolve() / name
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / "inference_config.json", {"weights_sha256": sha256(weights),
               "source": str(source), "confidence": confidence, "imgsz": 640,
               "names": model.names, "device": device,
               "note": "Per-frame detections, not confirmed fire events or geographic coordinates"})
    count = 0
    with (target / "detections.jsonl").open("w", encoding="utf-8") as handle:
        for count, result in enumerate(model.predict(str(source), stream=True, conf=confidence,
                imgsz=640, device=device, save=save_video, project=str(target.parent),
                name=target.name, exist_ok=True, verbose=False), 1):
            detections = []
            for box in result.boxes:
                cls = int(box.cls.item())
                detections.append({"class_id": cls, "label": model.names[cls],
                                   "score": float(box.conf.item()), "xyxy": box.xyxy[0].tolist()})
            handle.write(json.dumps({"frame_index": count-1, "image_shape": list(result.orig_shape),
                                     "detections": detections}, ensure_ascii=False) + "\n")
    print(f"Saved {count} frames: {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    predict(args.weights, args.source, args.output, args.confidence, args.device, args.save_video)
