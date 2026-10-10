# 두 사람의 D-Fire 학습 준비와 실행

기준 모델은 YOLO11n 하나다. 학습을 시작하기 전 각자 사본 노트북에서 GPU·공유 경로·데이터·라벨·백업 경로를 확인한다. 기본 설정은 학습 OFF이며 준비 코드를 실행해도 80epoch는 시작되지 않는다.

## 2026-09-30 validator 수정 반영

`training_workflow.py`의 평가 결과 폴더 접근 오류를 수정했다. 새 노트북의 준비 셀 출력에 `Workflow: 2026-09-30-validator-fix-1`이 보여야 한다.

이미 자기 Drive에 노트북 사본을 만든 경우 공유 원본의 변경이 자동 반영되지는 않는다. 학습 전이므로 공유 폴더 동기화 완료 후 최신 `dfire_team_training.ipynb`를 다시 사본 저장하고, 각자의 MEMBER_ID와 DRIVE_ROOT를 재입력하는 방법을 권장한다. 새 사본에서 위부터 준비 셀을 실행하면 된다. 새 노트북은 준비 셀을 재실행할 때도 예전 Python 모듈 캐시를 제거한다.

실제 YOLO11 CPU 평가·결과 복사·JSON 저장 통합 검사와 회귀 테스트 18개가 통과했다. Colab GPU에서의 본 학습은 아직 실행하지 않았다. 전체 점검 내용은 [PRETRAINING_CHECKS.md](PRETRAINING_CHECKS.md)를 확인한다.

## 지금 사용할 파일

- 노트북: `notebooks/dfire_team_training.ipynb`
- 학습 설정: `configs/train_yolo11n_v1.json`
- 모델 선정 이유: `docs/MODEL_DECISION.md`
- Colab 패키지: `requirements-colab.txt`
- 이전 `dfire_colab_pipeline.ipynb`는 YOLO26 시험 당시의 보관본이다. 새 본 학습에는 사용하지 않는다.

## 각자 수동으로 할 준비

1. Google Drive 웹에서 공유 프로젝트의 편집 권한과 `Data/DFire` 안의 원본 ZIP 두 개가 보이는지 확인한다.
2. 팀원 계정에서 공유 폴더가 '공유 문서함'에만 있으면 '내 드라이브에 바로가기 추가'를 한다. 노트북의 `DRIVE_ROOT`를 실제 바로가기 경로에 맞춘다. Windows의 G: 경로와 Colab의 `/content/drive/MyDrive` 경로는 다르다.
3. `Code/fire-ai/notebooks/dfire_team_training.ipynb`를 Colab으로 열고 **각자 자기 Drive에 사본 저장**한다. 한 공유 노트북의 MEMBER_ID를 서로 덮어쓰지 않는다.
4. 런타임 유형을 GPU로 변경한다. 각자 자기 계정의 독립 런타임 하나를 사용한다. 할당 가능한 GPU와 이용시간은 계정별로 달라질 수 있다.
5. 설정 셀에서 본인은 `MEMBER_ID='jongwon'`, 팀원은 `MEMBER_ID='member2'`처럼 다르게 설정한다. `BATCH=16`, 실행 스위치는 모두 False로 둔다.
6. 준비 셀 1~5를 순서대로 실행한다. GPU가 없거나 데이터/공유 경로/Drive 쓰기 검사가 실패하면 본 학습에 들어가지 않는다.
7. smoke·fire·배경 라벨 예시를 직접 확인한다. 정상이라면 라벨 확인 셀의 `LABELS_REVIEWED=True`로 변경하고 그 셀을 실행한다.
8. **사전 준비만 하는 지금은 여기까지.** Colab 런타임이 나중에 초기화되면 로컬 준비를 다시 해야 한다.

현재 노트북의 CPU 개발환경을 다시 설치할 필요는 없다. Colab에서는 `requirements-colab.txt`만 설치해 제공된 CUDA용 PyTorch를 유지한다. Windows용 `requirements-lock.txt`를 Colab에 그대로 설치하지 않는다.

## 나중에 본 학습을 시작할 때

노트북의 '6. 여기까지가 사전 준비 완료' 아래 **본 학습 셀 자체**에서 `RUN_FULL_TRAIN=True`로 바꾸고 실행한다. 위 설정 셀만 바꾸면 본 학습 셀이 다시 False로 덮어쓰므로 반드시 실제 본 학습 셀에서 변경한다. 다음 validation 셀도 실행하면 학습이 끝난 모델을 검증하고 Drive에 지표를 저장한다.

두 사람 모두 같은 모델과 분할·seed로 실행한다. 서로 다른 모델을 비교하는 실험이 아니다. 각자 80epoch를 돌려도 두 파일이 자동으로 합쳐지거나 시간이 절반으로 줄어들지 않는다.

- 시간은 처음 2~3epoch의 실제 로그로 다시 추정한다. 과거 YOLO26n 시험은 약 18분/3epoch였지만 새 모델·분할·GPU에서는 달라진다.
- GPU 메모리 부족이면 새 실행에서 `BATCH=8`로 낮춘다. 가능하면 둘 다 같은 batch로 맞춘다. 기존 실행을 정확히 재개할 때는 원래 batch를 유지한다.
- 조기 종료되면 80epoch 미만이어도 정상일 수 있다. 성능이 충분하다는 보장은 아니며 결과를 검토한다.

## 결과와 중간 백업 위치

공유 Drive에서:

```text
Data/DFire/
  D-Fire.zip                         원본 보존
  d-fire 텍스트 분할.zip              공식 분할 원본 보존
  pretrained/yolo11n.pt               공통 초기 가중치
  pretrained/yolo11n_manifest.json    가중치 해시와 로딩 점검
  preflight_reports/                 로컬 데이터 전체 검사 기록
  training_results/
    yolo26n_smoke_test_3ep/           이전 시험 결과 보존
    yolo11n_v1/
      jongwon_실행시각_고유번호/
      member2_실행시각_고유번호/
```

각 실행 폴더에는 다음이 저장된다.

- `run.json`, `environment.json`, `pip-freeze.txt`: 학습 조건과 실행환경.
- `source/`: 해당 실행의 코드·설정·노트북 사본.
- `dataset_summary.json`, `dataset_manifest_v001.csv`, `split_v001.csv`, `label_corrections.json`: 데이터 구성과 정리 기록.
- `checkpoints/epoch_0001/` 등: 매 epoch의 last.pt, best.pt, results.csv와 SHA-256 확인값.
- `latest_checkpoint.json`: 복사가 검증된 마지막 체크포인트 위치. 복사 실패 시 갱신하지 않는다.
- `status.json`: 초기화, 학습 중, 중단, 학습 완료, 검증 완료 상태.
- `final/best.pt`: 학습 완료 후 개발에 사용할 후보 모델. 기본 COCO 가중치와 혼동하지 않는다.
- `evaluations/`, `validation_summary.json`: 평가 결과와 클래스별 지표.

실제 학습은 Colab의 `/content`에서 수행한다. 원본 ZIP을 학습 중 계속 Drive에서 읽지 않는다. 백업용 모델·로그만 Drive로 복사한다. 매 epoch 사본을 보관하므로 실행당 여유 공간을 몇 GB 확보한다. 코드의 읽기 검증은 Google 서버 업로드 완료까지 보장하지 않으므로 실행 후 Drive 웹에서도 최종 파일을 확인한다.

## 학습이 끊겼을 때

1. 중단된 실행 폴더의 `latest_checkpoint.json`이 있는지 확인한다. 없으면 첫 epoch 백업이 완료되지 않은 것이므로 새로 시작한다.
2. Colab GPU 런타임에서 데이터와 패키지 준비 셀을 다시 실행한다.
3. `RESUME_FROM`에 **중단된 실행 폴더 전체 경로**를 입력한다. 예: `/content/drive/MyDrive/2026 3학년 2학기 진로탐색/Data/DFire/training_results/yolo11n_v1/jongwon_...`.
4. 원래 batch와 동일한 설정으로 라벨 확인과 본 학습 셀을 실행한다.
5. 새 실행 폴더에 부모 실행 경로를 기록하고, 마지막 완료 epoch 다음부터 진행한다. 이전 best.pt와 성능 로그도 복구한다.

오류로 불완전하게 남은 checkpoint 폴더는 완료 manifest와 일치하지 않으면 사용하지 않는다. 원래 실행과 재개 실행을 동시에 같은 복구 대상으로 조작하지 않는다. 중단 직전 epoch 전체는 다시 수행될 수 있으며, RNG 상태나 early stopping의 대기 카운터 차이 때문에 무중단 학습과 완전히 같은 결과는 보장하지 않는다.

## 라벨 정리 정책 v2

원본에서 폭·높이 0, 이미지 경계를 넘는 박스가 발견되었다. `prepare_dfire.py`는 압축을 푼 로컬 사본에만 동일한 기하학 규칙을 적용한다.

현재 원본에서 확인한 대상은 340개 라벨 파일의 397행이다. 18행은 면적 0, 379행은 이미지 경계 이탈이다. 개별 x/y/w/h 범위 검사만으로는 모든 경계 이탈을 찾을 수 없어 박스의 네 모서리도 검사했다.

- 면적이 0인 박스는 해당 라벨 행만 제외한다. 이미지와 나머지 박스는 유지한다.
- 이미지 밖으로 뻗은 박스는 이미지 영역 [0,1]과의 교집합으로 잘라 다시 YOLO 좌표로 저장한다.
- 이미지와 교집합이 없으면 해당 행을 제외한다.
- 완전히 중복된 박스 행은 한 개만 남긴다.
- 잘못된 클래스, NaN, 음수 크기는 자동 추정하지 않고 오류로 중단한다.
- 원래 행, 변경 후 행, 이유와 파일명을 모두 `label_corrections.json`에 남긴다.

공식 train/val/test **이미지 분할은 유지**한다. 위 규칙은 성능을 보기 전에 세 분할에 동일하게 적용한다. 결과는 '공식 fold 1 + 라벨 정리 v2' 성능으로 보고하고, 정리 전 공식 결과와 동일하다고 주장하지 않는다. 이전 3epoch 결과와 직접적인 수치 비교에도 주의한다. 기술 검사는 라벨 의미의 정확성이나 비슷한 영상 프레임 중복까지 보장하지 않는다.

## 두 사람 결과 공유와 최종 선택

노트북의 결과 확인 셀은 두 사람의 실행 폴더를 읽어 표로 표시한다. 환경·원본 해시·split 해시·정리된 라벨 manifest·초기 가중치가 같은지 확인한다. 숫자가 다르더라도 GPU 환경 차이부터 확인한다.

학습 완료 후 validation의 fire/smoke recall, precision, mAP와 정상 영상 오탐을 함께 검토한다. 실패 사례와 실행 제한사항을 기록하고 **한 실행의 best.pt**를 선정한다. `RUN_TO_EVALUATE`에 선정한 경로를 넣고 담당자 한 명만 `RUN_TEST_EVAL=True`, `FINAL_RUN_SELECTED=True`로 최종 시험을 수행한다. test를 보고 모델이나 임계값을 반복 조정하지 않는다.

## 모델을 실제 개발에 쓰는 방법

선정된 `final/best.pt`를 개발 PC의 `C:\fire-ai-local\weights\fire_v001.pt`로 복사한다. 기존 파일이 있으면 새로운 버전명으로 보관한다. 공유 프로젝트 폴더의 PowerShell에서:

```powershell
& "C:\fire-ai-local\.venv\Scripts\python.exe" "scripts\infer_media.py" --weights "C:\fire-ai-local\weights\fire_v001.pt" --source "C:\fire-ai-local\data\samples\sample.mp4" --output "C:\fire-ai-local\runs" --confidence 0.25 --device cpu --save-video
```

source는 본인이 보유한 실제 이미지/MP4 경로로 바꾼다. 0.25는 기능 연결용 초기값이며 운영 임계값은 validation으로 결정한다. CUDA 설정이 끝난 데스크탑에서는 `--device 0`을 사용할 수 있다.

추론은 프레임별 smoke/fire 클래스·점수·픽셀 박스를 `detections.jsonl`로 저장한다. 이 파일은 FastAPI가 반환할 데이터 구조의 출발점이다. React는 API 결과를 표시한다. 실제 백엔드에서는 모델을 요청마다 다시 불러오지 않고 작업 프로세스 시작 때 한 번 로드한다. 사건 판정·시간 필터·위경도 연결·확산 예측은 이후 단계이며 현재 스크립트가 확정 사건이나 지도 좌표를 만들어주지는 않는다.

## 완료 체크

- 사전 준비 완료: GPU/Drive 권한 통과, 분할 13,776/3,445/4,306, 라벨 그림 확인, 공통 초기 가중치 해시 일치.
- 학습 완료: final/best.pt와 로그·환경·데이터 기록을 Drive 웹에서 확인.
- 모델 1차 검증 완료: validation과 새 이미지/MP4의 추론 결과·실패 사례 확인.
- 최종 선정 완료: 팀에서 한 실행 선택, 최종 test 결과 기록, 개발 PC로 전달.

이번 자동화는 사전 준비와 검증을 수행한다. 80epoch 학습, Colab 계정 로그인·GPU 할당, 팀원의 공유 권한 확인은 아직 실행하지 않는다.
