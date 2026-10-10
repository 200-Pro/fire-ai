# 본 학습 모델 선정

2026-09-30 기준 모델은 **YOLO11n (`yolo11n.pt`)**이다. 두 팀원은 같은 모델을 사용하고, 서비스에는 최종 선정한 한 실행의 `best.pt`만 연결한다.

## 선정 이유와 한계

- 사용자의 우선순위는 안정성, D-Fire 활용, 수업 프로젝트에서의 구현 가능성이다.
- YOLO11은 2024년 공개되어 공식 학습·검증·추론·내보내기 경로를 제공한다. YOLO26도 지원되지만 더 새로운 구조와 학습 방식을 사용한다.
- D-Fire는 불꽃·연기에 대한 YOLO 형식 박스 라벨을 제공하므로 두 모델 모두 적용할 수 있다.
- 확인한 공식 자료에는 동일한 D-Fire 분할과 학습 조건에서 YOLO11 대 YOLO26의 우열을 확정할 근거가 없다. COCO 벤치마크를 D-Fire 성능으로 대체하지 않는다.
- 따라서 **YOLO11n 선택은 보수적인 개발 판단이며 D-Fire 정확도 우위가 입증됐다는 뜻이 아니다.** n 크기는 제한된 학습 시간과 영상 처리 부하를 고려한 기준선이다. 희미한 연기·작은 불꽃 성능은 실제 validation과 별도 영상으로 확인한다.
- 기존 YOLO26n 3epoch 결과는 환경 및 학습 흐름 시험 이력으로 보존한다. YOLO11 본 학습의 초기 가중치로 사용하지 않는다.

## 팀 실험 조건

모델 YOLO11n, Ultralytics 8.4.143, 공식 fold 1, 데이터 정리 정책 v2, seed 42, 640px, batch 16, 최대 80epoch, patience 15를 동일하게 사용한다. 두 실행은 결과 재현성을 확인하는 독립 실험이다. 같은 seed라도 GPU와 PyTorch/CUDA 환경 차이로 결과가 완전히 같지는 않을 수 있다.

두 사람이 완료한 모델을 합치지 않는다. validation 클래스별 recall·precision·mAP와 별도 영상 오탐을 보고 한 실행을 선택한다. test 평가는 선택 이후 담당자 한 명이 수행한다.

## 공식 자료

- YOLO11: https://docs.ultralytics.com/models/yolo11/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- D-Fire 원본과 라벨: https://github.com/gaia-solutions-on-demand/DFireDataset
- 학습과 재개: https://docs.ultralytics.com/modes/train/
- 체크포인트 콜백: https://docs.ultralytics.com/usage/callbacks/

## 문서 적용 순서

현재 실행은 `docs/TRAINING_READY.md`와 `notebooks/dfire_team_training.ipynb`를 따른다. 기존 실행가이드 PDF는 전체 프로젝트 참고자료이며 최신 체크포인트·경로 설정은 새 노트북을 기준으로 한다. 기획안·설계서의 'YOLO11 또는 YOLO26 중 하나 선정' 결정은 이 기록으로 구체화한다.
