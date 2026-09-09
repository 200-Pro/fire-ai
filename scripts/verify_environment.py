from pathlib import Path

import cv2
import torch
import ultralytics


LOCAL_ROOT = Path(r"C:\fire-ai-local")


def main() -> None:
    print(f"Python environment: {Path(__import__('sys').executable)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA enabled locally: {torch.cuda.is_available()}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    for name in ("data", "weights", "runs", "mlruns"):
        path = LOCAL_ROOT / name
        print(f"{name}: {path} ({'OK' if path.is_dir() else 'MISSING'})")


if __name__ == "__main__":
    main()

