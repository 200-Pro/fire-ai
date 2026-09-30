# 프로젝트 진행상황

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
