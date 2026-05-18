# AI Bot

Telegram에서 목표를 입력받아 여러 LLM 에이전트가 순차적으로 작업하는 Python 기반 멀티에이전트 봇입니다. Gemma/OpenAI 호환 Chat Completions 엔드포인트를 사용하며, 작업 상태는 SQLite에 저장하고 결과는 Markdown 파일로 남깁니다.

## 주요 기능

- Telegram 명령어 기반 목표 접수
- Leader, Planner, Executor, Reviewer 에이전트 파이프라인
- SQLite 기반 목표/태스크/지식 저장
- URL 또는 텍스트 지식 주입
- 선택적 e2b Python 샌드박스 실행
- 선택적 GitHub Contents API 자동 커밋
- 로컬 Markdown 결과 저장

## 프로젝트 구조

```text
.
├── main.py                    # 애플리케이션 진입점
├── scripts/
│   └── dryrun_llm.py          # Telegram 없이 LLM 연결 확인
├── src/
│   ├── agents/                # Leader, Planner, Executor, Reviewer
│   ├── bot/                   # Telegram 봇 핸들러
│   ├── knowledge/             # URL/텍스트 지식 주입
│   ├── llm/                   # Gemma/OpenAI 호환 클라이언트
│   ├── storage/               # 로컬 저장 및 GitHub 커밋
│   ├── task_queue/            # SQLite 작업 큐
│   └── tools/                 # e2b 실행 도구
└── tests/
    └── test_smoke.py          # 외부 호출 없는 핵심 smoke 테스트
```

## 요구 사항

- Python 3.11 이상 권장
- Telegram BotFather에서 발급한 봇 토큰
- OpenAI 호환 `/v1/chat/completions` 엔드포인트

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 환경 변수

`.env.example`을 참고해 `.env` 파일을 만듭니다.

```env
TELEGRAM_BOT_TOKEN=put-token-from-BotFather-here
GEMMA_BASE_URL=http://localhost:11434
GEMMA_MODEL=gemma-4
GEMMA_API_KEY=
DB_PATH=data/tasks.db
OUTPUT_DIR=data/outputs
E2B_API_KEY=
GITHUB_TOKEN=
GITHUB_REPO=owner/name
GITHUB_BRANCH=main
```

`.env`, `data/`, `*.db`는 Git에 포함하지 않습니다.

## 실행

```powershell
python main.py
```

Telegram에서 사용할 수 있는 주요 명령어:

```text
/start
/goal <목표>
/status
/result
/inject <URL 또는 텍스트>
/reset
```

## LLM 연결 확인

```powershell
python scripts\dryrun_llm.py
```

## 테스트

외부 API 호출 없이 핵심 로직을 확인합니다.

```powershell
python tests\test_smoke.py
```

`pytest`가 설치되어 있다면 다음 방식도 사용할 수 있습니다.

```powershell
python -m pytest -q
```

## 참고

`Modelfile.gemma3-16k`는 Ollama에서 `gemma3:4b` 기반 16K 컨텍스트 모델을 만들기 위한 예시입니다.
