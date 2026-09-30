# 팀원 로컬 설정 안내

공유 폴더의 코드만 함께 사용하고, 가상환경·데이터·모델·실행 결과는 각자 PC의 `C:\fire-ai-local`에 저장합니다.

## 팀원이 각자 해야 하는 작업

공유 프로젝트 폴더에서 PowerShell을 열고 다음을 실행합니다.

```powershell
Set-Location "G:\내 드라이브\2026 3학년 2학기 진로탐색\Code\fire-ai"
powershell -ExecutionPolicy Bypass -File ".\scripts\setup_local.ps1"
```

Google Drive의 드라이브 문자나 경로가 다르면 첫 번째 줄만 자신의 PC에 맞게 변경합니다.

설치 후 VS Code에서 다음 인터프리터를 선택합니다.

```text
C:\fire-ai-local\.venv\Scripts\python.exe
```

선택 순서는 `Ctrl+Shift+P` → `Python Select Interpreter` → `Enter interpreter path`입니다.

환경과 공유 파일을 한 번에 점검하려면 프로젝트 폴더에서 다음을 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\audit_system.ps1"
```

## 한 명만 해야 하는 작업

- 공유 폴더와 Git 저장소 최초 생성
- `.gitignore`, README, 공통 설정 파일 관리
- GitHub 원격 저장소 생성 및 연결
- 공공 API 신청 담당과 승인 상태 기록
- Colab 학습 노트북과 최종 `best.pt` 관리

## Colab에서 실행할 첫 셀

```python
!pip install -U ultralytics
```

Colab 학습 모델명은 `yolo26n.pt`를 사용합니다. 학습 완료 후 생성된 `best.pt`를 로컬 추론 PC의 `C:\fire-ai-local\weights`에 저장합니다.

## 주의사항

- `.env`, 원본 데이터, `*.pt`, `runs`는 Git에 올리지 않습니다.
- D-Fire 클래스 번호는 실제 배포본의 설정과 라벨을 확인한 뒤 확정합니다.
- 현재 공통 설정은 `0=smoke`, `1=fire`이지만 확인 전 임의로 라벨 번호를 변환하지 않습니다.
- 공유 Drive 폴더에서 대용량 학습이나 반복 추론을 실행하지 않습니다.
