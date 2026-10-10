# 프로젝트 진행상황

## 2026-09-30 본 학습 준비 갱신

- 추가 수정: validator 결과 경로 오류 해결, 실제 라벨/YAML 정합성 검사와 재개 보호·Colab 코드 캐시 갱신 추가. 회귀 테스트 18개와 실제 YOLO11 CPU 평가/추론 통합 검사 통과. 현재 코드 버전은 `2026-09-30-validator-fix-1`이며 실제 본 학습은 아직 미실행.

- 본 학습 모델은 YOLO11n으로 선정. 기존 YOLO26n 3epoch 결과는 시험 이력으로 보존.
- 최신 실행 파일: `notebooks/dfire_team_training.ipynb`, 상세 안내: `docs/TRAINING_READY.md`.
- 개인별 고유 실행 폴더, 매 epoch 체크포인트 해시 검증 백업, 중단 재개, 검증 지표 공유, test 실행 확인, 이미지/MP4 추론 코드 추가.
- 원본 전수 재검사에서 폭/높이 0 또는 정규화 범위 위반 라벨 26행 발견. 기존 '라벨 검증 완료' 기록은 엄밀하지 않았음. 원본 ZIP은 유지하고 학습용 사본에 v2 좌표 정리 정책 적용 및 전체 변경 로그 보관.
- 경계 좌표까지 검사한 전체 정리 대상은 340개 라벨 파일의 397행: 면적 0인 박스 18행 제외, 경계 이탈 379행 클리핑. 공식 이미지 분할은 유지하며 test에도 같은 사전 규칙 적용.
- 모든 학습 실행 스위치는 False. 새 모델의 80epoch 학습과 Colab GPU 실행은 아직 수행하지 않음.
- 로컬 전수 이미지 디코딩 21,527장, YOLO11n 실제 CPU 추론, 모의 백업/재개 등 9개 테스트 통과. 증빙과 검증 한계: [PRETRAINING_CHECKS.md](PRETRAINING_CHECKS.md).

아래는 이번 갱신 이전의 상태와 시험 결과 기록이다. 현재 실행 방법은 위 최신 안내를 따른다.

기준일은 2026년 9월 30일이다. 현재 개발환경, 공개 데이터 확보, D-Fire 검증, YOLO26n 3epoch 시험 학습까지 완료했다. 계획상 5주차에 해당하지만 정식 학습과 오류분석에 필요한 재현 파일은 아직 보완 중이다.

## 완료

- 노트북 CPU 개발환경과 VS Code 설정
- GitHub Organization 저장소 연결
- 기상청 단기예보와 산불위험예보 API 신청 및 승인
- D-Fire 원본 및 공식 분할 파일 확보
- 이미지 21,527개와 라벨 21,527개 검증
- 클래스 0 smoke, 클래스 1 fire 확인
- Colab GPU에서 YOLO26n 3epoch 시험 학습
- best.pt, args.yaml, results.csv, confusion matrix 저장

## 현재 성능

- Precision 0.6683
- Recall 0.6068
- mAP50 0.6621
- mAP50-95 0.3614

이 수치는 3epoch 파이프라인 시험 결과이며 최종 모델 성능이 아니다.

## 자동화 추가

- `scripts/prepare_dfire.py`: 압축 해제, 공식 fold 적용, 파일 짝 검사, 라벨 검사, manifest와 YAML 생성
- `notebooks/dfire_colab_pipeline.ipynb`: Colab 데이터 준비, 시각화, 시험 학습, 본 학습, test 평가 순서
- `configs/fire_v001_colab.yaml`: Colab 기준 데이터 경로
- `docs/label_guide_v1.0.md`: 클래스와 라벨 QA 규칙

## 남은 핵심 작업

1. Colab 노트북을 실행해 자동 생성된 manifest와 split 결과를 보관한다.
2. 학습에 사용하지 않은 표본의 라벨 시각화를 확인한다.
3. 80epoch 본 학습을 실행하고 early stopping 결과를 저장한다.
4. 클래스별 FP와 FN 사례를 모아 error analysis를 작성한다.
5. 고정 test 세트는 모델 선택 완료 후 한 번 평가한다.
6. MP4 추론, 시간 필터, FP/hour, 탐지 지연, FPS를 구현한다.
7. 모델 확정 후 ONNX로 내보내고 PyTorch 결과와 비교한다.

## 보류

- API 실제 호출과 서비스 연동
- 실시간 CCTV 또는 휴대폰 카메라 입력
- 확산예측, 웹 지도, 데이터베이스, 배포
