# K-Flower Card Developer Guide

## 1) 목적
이 문서는 현재 코드베이스의 디렉토리 구조와 파일 역할을 고정하기 위한 개발 기준서다.  
새 파일 추가, 경로 변경, 리팩터링 시 이 문서를 기준으로 판단한다.

## 2) 현재 구조 (기준)
```text
src/
  main.jsx
  App.jsx
  cards.js
  ai/
    aiPlay.js
    heuristicPolicyEngine.js
    modelPolicyEngine.js
    moePolicy.js
    policies.js
  app/
    useAiRuntime.js
    useReplayController.js
  engine/
    index.js
    runner.js
    state.js
    finalizeTurn.js
    rules.js
    scoring.js
    economy.js
    matching.js
    opening.js
    resolution.js
    turnFlow.js
    capturesEvents.js
    combos.js
  heuristics/
    heuristicV3.js
    heuristicV4.js
  ui/
    components/
    i18n/
    utils/
```

## 3) 레이어 역할

### Entry / 화면
- `src/main.jsx`: React 진입점.
- `src/App.jsx`: 화면 상태, 이벤트 핸들링, 엔진/AI 호출 오케스트레이션.

### App Hooks
- `src/app/useAiRuntime.js`: AI 모델 로딩, 실행 옵션 선택, 자동 턴 실행.
- `src/app/useReplayController.js`: 리플레이 상태/재생 제어.

### AI
- `src/ai/policies.js`: 정책 ID, 모델 카탈로그, 정책 선택 유틸의 단일 기준점.
- `src/ai/aiPlay.js`: AI 실행 단일 진입점. 모델/휴리스틱 분기.
- `src/ai/modelPolicyEngine.js`: 정책 모델 inference/행동 선택.
- `src/ai/heuristicPolicyEngine.js`: 휴리스틱 기반 행동 선택(v3/v4 호출).
- `src/ai/moePolicy.js`: 공격/방어(MoE) 선택 규칙.

### Engine
- `src/engine/index.js`: 엔진 공개 API 배럴(외부에서 우선 사용).
- `src/engine/runner.js`: 자동 턴 진행(`getActionPlayerKey`, `advanceAutoTurns`).
- `src/engine/state.js`: 게임 상태머신 본체(초기화, 턴 진행, 의사결정 액션).
- `src/engine/finalizeTurn.js`: 턴 종료 후 정산/정규화/Go-Stop 연계 핵심 로직.
- `src/engine/rules.js`: 룰셋 상수.
- `src/engine/scoring.js`: 점수 계산.
- `src/engine/economy.js`: 골드/정산 계산.
- `src/engine/matching.js`: 월 매칭 판정.
- `src/engine/opening.js`: 초반 세팅/대통령 판정.
- `src/engine/resolution.js`: 라운드 종료 판정/결과 확정.
- `src/engine/turnFlow.js`: 턴 플로우 유틸(패스카드/리빌 만료 등).
- `src/engine/capturesEvents.js`: 캡처/강탈/이벤트 유틸.
- `src/engine/combos.js`: 족보 관련 공통 상수/유틸.

### Heuristics
- `src/heuristics/heuristicV3.js`: v3 규칙.
- `src/heuristics/heuristicV4.js`: v4 규칙.

### UI
- `src/ui/components/*`: 화면 컴포넌트.
- `src/ui/i18n/i18n.js`: 다국어.
- `src/ui/utils/*`: UI 공통 유틸(리플레이/카드 정렬 등).

### Shared Domain
- `src/cards.js`: 카드 덱 정의/에셋 경로/셔플.

## 4) 의존성 규정 (중요)
- `ui/*`는 `engine/*` 내부 구현 파일을 직접 참조하지 않는다.  
  기본적으로 `engine/index.js`, `engine/runner.js`, `ai/*` 공개 API만 사용한다.
- `ai/*`는 `ui/*`를 import하지 않는다.
- `engine/*`는 `ai/*`, `ui/*`, `app/*`를 import하지 않는다.
- 휴리스틱 구현(`heuristics/*`)은 React/UI 의존을 두지 않는다.
- 구조 단순화를 위해 루트 래퍼 파일(`src/gameEngine.js` 등)은 재도입하지 않는다.

## 5) 파일 추가/분리 기준
- 파일 수 증가를 최소화한다.
- 먼저 기존 파일에 합칠 수 있는지 검토한다.
- 아래를 모두 만족할 때만 새 파일을 만든다.
  - 역할이 독립적이고 재사용 경계가 분명함
  - 순환 참조 없이 분리 가능함
  - 테스트/시뮬레이터/UI에서 공통으로 쓰일 가능성이 있음

## 6) 변경 체크리스트
- 경로 변경 시 `src`와 `scripts`의 import 경로를 함께 수정했는가
- `npm run build`가 통과하는가
- 구조 규정(4번)을 위반하는 신규 import가 없는가

## 7) 현재 권장 진입점
- 엔진 기능 사용: `src/engine/index.js`
- 턴 진행 유틸 사용: `src/engine/runner.js`
- 정책/모델 목록 사용: `src/ai/policies.js`
