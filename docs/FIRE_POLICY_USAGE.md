# 오탐 억제 후처리 사용 방법

이번 작업은 best.pt 재학습이 아니라 **모델 예측 뒤의 필터**를 추가한 것이다. 원래 학습 노트북과 infer_media.py의 기존 실행은 자동으로 바뀌지 않는다. 아래 새 실행 코드를 사용해야 후처리가 적용된다.

## 구성

- `configs/fire_policy_v1.json`: 검증된 모델 해시, 입력 해상도/예측 조건, 후처리 조건. 모델 파일이 바뀌면 현재 설정은 다시 검증해야 한다.
- `scripts/fire_policy.py`: 프레임 예측의 fire/smoke 박스를 필터링한다. API 프로그램에서도 `filter_predictions(predictions, policy, (height, width))`를 재사용할 수 있다.
- `scripts/infer_fire_frame.py`: JPG/PNG 등 이미지 1장을 모델로 추론하고 후처리된 사진/JSON을 저장한다.
- `scripts/tune_fire_policy.py`: **validation만** 사용해 저장 예측에서 후보 조건을 비교한다. test가 입력되면 거부한다.
- `scripts/compare_fire_resolution.py`: validation만 로컬에 준비하고 동일 모델의 640/960 CPU 추론을 비교한다. 모델과 원본 ZIP을 수정하지 않는다.

## 필터의 의미

기본 후보 수집은 conf 0.05, 원래 YOLO NMS IoU 0.7로 한다. 최종 표시 기준은 설정 파일의 filter가 결정한다. 이는 모든 0.05 후보를 화재로 표시한다는 뜻이 아니다.

현재 640 후보 설정은 fire 0.25 이상을 유지하고, 0.22~0.25의 약한 fire 후보는 **높은 점수 fire가 근처에 있을 때만** 허용한다. 약한 fire만 있는 새 장면에서는 경보를 새로 만들지 않는다. 근접 기준은 중심 간 x/y 거리를 각각 이미지 너비/높이로 정규화한 거리 0.4 이하다.

추가 중복 제거 IoU는 0.45이며 fire끼리만 적용한다. smoke는 원래 0.25 기준을 유지한다. 이 두 번째 중복 제거를 'YOLO NMS를 바로 0.45로 바꾼 것'과 동일하게 해석하면 안 된다. raw Predict의 IoU 0.7과 추가 필터 0.45를 둘 다 유지해야 검증 조건을 재현한다.

## 로컬 PowerShell 실행

프로젝트 폴더로 이동하고 이미지 경로만 바꾼다. 아래는 이번에 새 로컬 사본으로 준비한 validation 이미지 예시다.

```powershell
Set-Location "G:\내 드라이브\2026 3학년 2학기 진로탐색\Code\fire-ai"
& "C:\fire-ai-local\.venv\Scripts\python.exe" "scripts\infer_fire_frame.py" `
  --weights "G:\내 드라이브\2026 3학년 2학기 진로탐색\Data\DFire\training_results\yolo11n_v1\jongwon_to70_20261004T144737Z_9154ce3d\final\best.pt" `
  --source "C:\fire-ai-local\data\val_resolution_20261005_crc\val\images\AoF04266.jpg" `
  --output "C:\fire-ai-local\runs\fire-policy-frames"
```

출력은 고유 폴더 아래 `annotated.jpg`와 `detections.json`이다. 실제 출력 사진의 박스와 JSON은 **필터 후 결과**다. `fire_candidate=True`는 해당 프레임의 후보이며 화재 사건 확정이 아니다. MP4는 이 명령에서 받지 않는다.

팀원 PC에는 위 C: 이미지 사본이 자동으로 생기지 않는다. 팀원은 본인의 테스트 이미지 경로를 사용하거나 Colab에서 실행한다. 가상환경/모델 다운로드 절차는 기존 안내를 따른다.

## 기존 Colab 분석 런타임에서 한 장 실행

기존 불꽃 미탐 분석 노트북에서 Drive 마운트와 데이터 준비가 끝난 상태라면 새 코드 셀에 아래를 넣을 수 있다. 처음부터 학습할 필요는 없다. GPU 패키지는 기존 런타임을 유지한다.

```python
import sys, shutil, uuid, importlib
from pathlib import Path
from IPython.display import display, Image

ROOT = Path('/content/drive/MyDrive/2026 3학년 2학기 진로탐색')
PROJECT = ROOT / 'Code' / 'fire-ai'
RUN = ROOT / 'Data' / 'DFire' / 'training_results' / 'yolo11n_v1' / 'jongwon_to70_20261004T144737Z_9154ce3d'
DATA = Path('/content/fire-ai-data-v2')  # 현재 런타임에서 준비한 경로

CODE = Path('/content') / ('fire-policy-code-' + uuid.uuid4().hex[:8])
shutil.copytree(PROJECT / 'scripts', CODE)
sys.path.insert(0, str(CODE))
for name in ('training_workflow', 'analyze_fire_misses', 'fire_policy', 'infer_fire_frame'):
    sys.modules.pop(name, None)
importlib.invalidate_caches()
from infer_fire_frame import predict_frame

out = predict_frame(
    RUN / 'final' / 'best.pt',
    DATA / 'val' / 'images' / 'AoF04266.jpg',  # 파일명만 바꿔서 다른 사진 확인
    Path('/content/fire-policy-frames'),
    PROJECT / 'configs' / 'fire_policy_v1.json',
    device='0',
)
display(Image(filename=str(out / 'annotated.jpg')))
```

새 런타임이면 기존 분석 노트북의 Drive·패키지·데이터 준비 셀을 먼저 실행한다. 결과는 /content에만 저장되므로 런타임 종료 시 사라질 수 있다. 필요한 결과만 Drive에 별도 보관한다.

## 주의

FP 감소는 박스 평가 기준의 결과다. 불꽃 없는 이미지에서 잘못 울리는 경보가 줄었는지는 별도 이미지 지표를 봐야 한다. 개별 박스의 검출/미탐은 바뀔 수 있다. 저장 validation을 반복 비교했으므로 독립 최종 평가나 실제 CCTV 성능 보장은 아니다. 원래 test 점수는 기존 모델의 기록으로 유지한다.
