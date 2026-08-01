# STRING Chrome Extension

## 설치
1. 백엔드를 실행합니다: `..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8765`
2. `chrome://extensions`에서 개발자 모드를 켭니다.
3. `압축해제된 확장 프로그램을 로드`를 눌러 이 `browser-extension` 폴더를 선택합니다.

## 사용
웹페이지에서 텍스트를 선택한 뒤 floating `Analyze with STRING` 버튼 또는 우클릭 메뉴를 누릅니다. Side panel에서 입력과 설정을 확인하고 실행하면 `/api/discussion-lab/run`으로 최소 request body를 전송합니다. 성공하면 실제 `run_id`가 포함된 `/discussion-lab?run_id=...`를 엽니다.

설정은 확장 프로그램의 Options에서 endpoint, 결과 URL, model, reasoning effort, 토큰 제한, timeout을 변경할 수 있습니다. `chrome://`, 확장 프로그램 페이지 등 content script가 실행되지 않는 페이지에서는 우클릭/버튼을 사용할 수 없습니다.

## 검증
`node --check`로 JS 문법을 확인하고, `manifest.json`을 JSON으로 파싱해 경로를 확인합니다. API key는 확장 프로그램에 저장하거나 출력하지 않습니다.
