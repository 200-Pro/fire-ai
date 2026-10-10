# 본 학습 전 자동 점검 결과

점검일: 2026-09-30. 모델 학습은 실행하지 않았다.

## validator 수정 후 추가 정합성 검사

코드 버전: `2026-09-30-validator-fix-1`.

- `model.validator.save_dir` 참조 두 곳을 제거하고 `on_val_end` 콜백에서 실제 출력 폴더를 받도록 수정.
- 메타데이터 해시뿐 아니라 실제 라벨 파일 해시·YAML 경로·클래스·분할 목록·파일 목록도 학습/평가 전에 검사.
- 재개할 파일에 optimizer/epoch가 없으면 새 학습으로 조용히 전환되지 않도록 중단. 이미 목표 epoch에 도달한 체크포인트도 재학습 방지. 존재하는 옛 YAML로 잘못 재개하는 경우도 차단.
- Drive 상태 기록까지 실패하더라도 원래 학습/백업 오류가 가려지지 않도록 보완.
- Colab 준비 셀 재실행 시 캐시된 이전 프로젝트 모듈을 제거하고 새 로컬 코드 사본에서 다시 import. 화면에 Workflow 버전을 출력.
- 회귀 테스트 **18개 통과**. 실제 validator 속성이 없는 모의 객체로 평가 분기까지 검증.
- 실제 Ultralytics 8.4.143 / YOLO11n / CPU를 사용한 통합 검사 통과: 합성 이미지 validation → 실제 콜백 → 그래프 복사 → metrics JSON → 공유 결과 조회 → 이미지 추론 JSONL.
- 통합 검사는 무학습·임의 초기 가중치·임시 합성 데이터만 사용. 0으로 나온 성능 수치는 의미가 없으며 프로젝트 성능으로 보고하지 않는다. 원본 D-Fire와 기존 best.pt를 사용하거나 변경하지 않았다.
- Python 파일 11개와 노트북 22개 셀 구문 확인. 학습/test 실행 스위치는 False 유지.

재현 명령(프로젝트 폴더):

```powershell
& "C:\fire-ai-local\.venv\Scripts\python.exe" -m unittest discover -s tests -v
& "C:\fire-ai-local\.venv\Scripts\python.exe" tests/check_real_validation.py
```

이전 설명에서 `metrics.save_dir`를 사용할 수 없다고 단정한 부분은 정정한다. 현재 DetectionValidator는 실제 평가가 끝날 때 metrics.save_dir를 동적으로 추가한다. 생성 직후 DetMetrics만 검사해서는 확인할 수 없는 동작이다. 수정 코드에서는 실제 validator 콜백을 통해 폴더를 확보한다.

아래 9개 테스트 통과 기록은 이전 점검 시점의 기록이다. 현재는 위 18개 테스트와 실제 CPU 통합 검사를 기준으로 한다. Colab GPU 본 학습·실제 optimizer 학습 재개는 여전히 미실행이며 오류가 전혀 없다는 보장은 아니다.

## 실제로 확인한 항목

- 원본 D-Fire 이미지 21,527장 전부 디코딩 성공.
- 공식 fold 1: train 13,776 / validation 3,445 / test 4,306. 파일명 분할 중복 없음.
- 이미지·라벨 짝과 클래스 확인. 배경 이미지 9,838장.
- 원본 라벨 정리 대상: 340개 파일의 397행. 면적 0 박스 18행 제외, 이미지 경계 밖 박스 379행 클리핑 대상.
- 정리 후 예상 박스: smoke 11,854 / fire 14,685. 원본 라벨의 클래스는 변경하지 않음.
- 원본 ZIP은 보존. 변경 규칙은 prepare_dfire.py가 Colab에 푼 사본에만 적용.
- YOLO11n 공식 사전학습 가중치 다운로드·로딩·640px 빈 이미지 CPU 추론 성공. 이는 실행 호환성 검사이며 화재 탐지 성능 검사가 아님.
- 체크포인트 복사 실패 시 이전 포인터 보존, 손상 파일 재개 거부, 과거 best 복구, 고유 실행 ID, 학습/test 기본 비활성, 라벨 정리와 준비 재실행, 모의 학습기 최종 저장 등을 포함한 9개 unittest 통과.
- 새 노트북 22개 셀의 Python 구문 검사, 실행 스위치 기본 False 확인.

## 검사 증빙

- 데이터 전수 검사: `Data/DFire/preflight_reports/local_20260930T075310Z_266a99f3.json` (프로젝트 최상위 기준).
- 초기 가중치 검사: `Data/DFire/pretrained/yolo11n_manifest.json`.
- 원본 데이터 SHA-256: `3824fb3ce32cfa8b538792dfd460603648d271072ac6ae34f1af0c713f60260c`.
- 분할 ZIP SHA-256: `7eac0b41e3b8c9aae9488d2c2352e1b39d9d5833675e8b51a434be738b43a33e`.
- 초기 YOLO11n SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.

## 각자의 Colab에서 아직 확인해야 할 항목

Google 계정 로그인, 공유 폴더 권한, GPU 할당, Colab 패키지 설치, 전체 데이터 준비 코드의 실제 Colab 실행, 라벨 의미의 시각 검토, 장시간 학습과 실제 GPU 체크포인트 재개. 로컬 파일 복사/모의 테스트는 이 환경 검증을 대체하지 않는다.

현재 로컬은 Python 3.12.10, torch 2.14.0+cpu, Ultralytics 8.4.143이며 CUDA가 없다. Colab GPU 학습의 결과 성능은 아직 없다. 본 학습 전환은 `docs/TRAINING_READY.md`를 따른다.
