# 학습 코드·서비스 인수인계 — 2026-10-10

## 저장소를 구분하기

이 저장소 `200-Pro/fire-ai`는 D-Fire 준비·Colab 학습·평가·미탐 분석·후처리를 담당한다. 서비스는 [200-Pro/fire-detection](https://github.com/200-Pro/fire-detection)이며 실제 개발 위치는 `D:\fire-dettection`이다.

전체 단계별 기록, 초기 기술스택에서 바뀐 항목, 실패·검증·다음 작업은 [서비스 기술 아카이브](https://github.com/200-Pro/fire-detection/tree/main/docs/history)와 [새 프로젝트 인수인계](https://github.com/200-Pro/fire-detection/blob/main/docs/history/07_HANDOFF.md)를 먼저 읽는다. 기존 PROJECT_STATUS·TRAINING_READY는 작성 당시 기록도 포함한다.

## 현재 모델·데이터

- 실제 클래스: `0=smoke`, `1=fire`. 초기 PDF의 반대 순서 예시는 사용하지 않는다.
- 공식 fold 1: train 13,776 / val 3,445 / test 4,306. 파일명 겹침·누락 0이며 사건 단위 누수 증명은 아니다.
- 라벨 v2는 사본에서 340파일·397행 정리: 0면적 18행 제외, 경계 클리핑 379행. 원본 ZIP은 보존했다.
- YOLO26n 3epoch 시험 → YOLO11n 80epoch 계획 → 69 완료·70 도중 중단 → 사용자 선택으로 70 완료.
- 선정 모델 SHA: `57c133a489976dc2f7341381b12cf160622e0bbaff33aeffdee4926e2093611d`.
- 기본 모델 test 전체 mAP50 0.769273 / mAP50-95 0.442155. fire recall 0.624739이며 실제 CCTV 사건 recall은 아직 측정하지 않았다.
- 동일 CPU 후처리 비교 FP 870→738, FN 639→638. 미탐 대폭 해결이나 실제 경보 15.2% 감소가 아니다. 불꽃 없는 이미지의 잘못된 후보 34장은 그대로다.
- 70epoch 이후 새 학습은 없다. 정책은 validation 개발 결과이며 새 독립 CCTV 정답 평가가 필요하다.

## 실행·검사

`notebooks/dfire_team_training.ipynb`가 본 학습, `dfire_fire_miss_analysis.ipynb`가 미탐 분석이다. 초기 `dfire_colab_pipeline.ipynb`는 YOLO26 시험 이력으로 보존한다. 학습·최종 test의 스위치는 기본 OFF다.

```powershell
cd 'G:\내 드라이브\2026 3학년 2학기 진로탐색\Code\fire-ai'
& 'C:\fire-ai-local\.venv\Scripts\python.exe' -m unittest discover -s tests -p 'test_*.py' -v
```

이번 업로드 전 35개가 통과했다. pytest가 아니라 unittest 방식이다. Colab은 기존 CUDA torch를 유지하고 `requirements-colab.txt`의 고정 Ultralytics를 사용한다. Windows CPU lock을 Colab에 설치하지 않는다.

새 학습은 데이터·split·라벨·초기 모델·환경 hash와 별도 run ID를 만든 뒤 시행한다. test로 threshold를 반복 선택하지 않는다. 모델 교체는 클래스·입출력·정책·SHA·회귀 예제·서버 재시작·rollback을 함께 확인한다.

## 공개 범위

코드·노트북·설정·회귀 검사·비밀 없는 Markdown/JSON 집계는 공개한다. 이미지 예시·원본 ZIP·pt·API 키·전체 개인 대화는 제외한다. 예전 PHOTO_REVIEW·REPORT의 사진 상대 링크가 GitHub에서 열리지 않을 수 있으며 사진은 권한 있는 팀 Drive·로컬에서 확인한다.

이 문서 작성은 새 학습·실제 CCTV 수집·프로젝트 이동을 수행하지 않았다.
