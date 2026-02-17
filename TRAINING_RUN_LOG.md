# TRAINING RUN LOG

## 2026-02-17 문서 동기화

- `TRAINING_RUN_LOG.md` 중요턴 트리거를 10개 그룹 기준으로 갱신
- `turn_end + option 의사결정 시점` 동시 기록 방식 반영
- 중요턴 기반 컨텍스트 보존/중복 제거 규칙 반영

## 1) `decision_trace` 기록 시점(핵심)

`decision_trace`는 턴마다 무조건 저장하지 않고, 아래 기준을 통과한 턴만 기록한다.

- 수집 기준 턴:
  - `kibo.type === "turn_end"`
  - 시뮬레이션 루프에서 `selectionPool.decisionType === "option"`인 별도 의사결정 이벤트
- 턴 확정 조건:
  - 루프에서 `nextTurnSeq > prevTurnSeq`일 때 before/after context 확정
- 사이드 필터:
  - 기본: 양쪽 턴 기록
  - `--trace-my-turn-only`: `mySide` 턴만 기록
- 후보 수 필터:
  - `train`: `candidateCount >= 1`
  - `delta`: `candidateCount >= 2`
- 중요턴 필터:
  - 기본 ON(`--trace-important-only`)
  - OFF 시 후보 수 조건만 통과하면 모두 기록

## 2) 중요턴 10개 트리거 상세

아래 10개 그룹으로 관리한다. (내부 구현상 세부 트리거로 분기될 수 있음)

1. `earlyTurnForced`
- 조건: 각 사이드 턴 카운트 `<= 1`
- 저장: `T`

2. `goStopOption`
- 조건: `decisionType === "option"` + `go/stop`
- 저장(기본): `T-1, T, T+1`
- 저장(옵션): `go` 선택 시 `--trace-go-stop-plus2` 활성화면 `T+2` 추가

3. `shakingYesOption`
- 조건: `actionType === "choose_shaking_yes"`
- 저장: `T, T+1`

4. `optionTurnOther`
- 조건: `president_stop/president_hold/five/junk/shaking_no` 등 기타 옵션
- 저장: `T, T+1`

5. `deckEmpty`
- 조건: `beforeDc.deckCount <= 0`
- 저장: `T-1, T`

6. `specialEvent`
- 조건: `matchEvents` 중 `eventTag !== "NORMAL"`
- `PPUK`: `T-1, T, T+1`
- `JJOB/DDADAK/PANSSEUL` 등 핵심 이벤트: `T-1, T`

7. `bombEvent`
- 조건: 폭탄 선언 또는 폭탄 카운트 변화
- 폭탄 선언(`declare_bomb`): `T-1, T, T+1`
- 카운트 변화만: `T, T+1`

8. `riskShift`
- 조건: `piBakRisk/gwangBakRisk/mongBakRisk` 변화
- 증가(0->1): `T-1, T`
- 감소(1->0): `T, T+1`
- 혼합 변화: `T`

9. `comboThreatEnter`
- 조건: 상대 족보 위협 상태 진입(`false -> true`)
- 조건(홍/청/초/고도리): 상대 진행 `>=2` && 내 블로킹 카드 `==0`
- 조건(광 1차): 상대 광 `==2` && 내 블로킹 광 `<=2`
- 조건(광 2차): 상대 광 `>=3` && 내 블로킹 광 `<=1`
- 저장: `T`

10. `terminalContext`
- 조건: 게임 종료 컨텍스트 보존
- 저장: 종료 기준 `2턴 전` 레코드 위치에 `terminalContext` 태그 부여 (`T`)

공통 규칙:
- 트리거별로 뽑힌 턴 인덱스는 게임 단위 `Set`으로 중복 제거 후 저장
- 중요턴이 하나도 없으면, 후보 조건을 통과한 턴을 그대로 반환
- `--trace-context-radius`는 기본 `0`이며 필요 시 전역 반경으로 추가 보존

## 3) `decision_trace` 턴 레코드 구조

`train` 모드 기본 레코드:

- `t`: 턴 번호
- `a`: 액터 사이드(`mySide`/`yourSide`)
- `o`: 순서(`first`/`second`)
- `dt`: 의사결정 타입(`play`/`match`/`option`)
- `cc`: 후보 수
- `at`: 실제 액션 타입
- `c`: 선택 카드 id
- `s`: 선택 바닥 카드 id
- `ir`: 중간보상(immediate reward)
- `dc`: 압축 decision context
- `tg`: 트리거 태그 배열(해당 시)

`dc` 필드:

- `phase`
- `deckCount`
- `handCountSelf`, `handCountOpp`
- `goCountSelf`, `goCountOpp`
- `carryOverMultiplier`
- `piBakRisk`, `gwangBakRisk`, `mongBakRisk`
- `oppJokboThreatProb`, `oppJokboOneAwayProb`, `oppGwangThreatProb`
- `currentScoreSelf`, `currentScoreOpp`

## 4) 게임 루트(1게임 1레코드)

- `game`, `seed`, `steps`, `completed`, `logMode`
- `firstAttackerSide`, `firstAttackerActor`
- `policy`
- `winner`, `score`
- `initialGoldMy`, `initialGoldYour`
- `finalGoldMy`, `finalGoldYour`
- `goldDeltaMy`, `goldDeltaYour`, `goldDeltaMyRatio`, `goldDeltaMyNorm`
- `gold`
- `decision_trace`

## 5) 검증 기록

문법 체크:

- `node --check scripts/selfplay_simulator.mjs` 통과
- `python -m py_compile scripts/run_parallel_selfplay.py scripts/01_train_policy.py scripts/02_train_value.py scripts/03_evaluate.py scripts/train_dual_net.py` 통과

100판 샘플 확인:

- 실행: `node scripts/selfplay_simulator.mjs 100 logs/_tmp_sync_100.jsonl --policy-my-side heuristic_v3 --policy-your-side heuristic_v3 --log-mode train --trace-all-turns`
- 결과: `terminalContext` 태그가 게임 수와 동일하게 기록됨
  - games: `100`
  - terminal hits: `100`
  - missing games: `0`

## 6) 관련 파일

- `scripts/selfplay_simulator.mjs`
- `scripts/run_parallel_selfplay.py`
- `scripts/01_train_policy.py`
- `scripts/02_train_value.py`
- `scripts/03_evaluate.py`
- `scripts/train_dual_net.py`
- `src/bot.js`
- `src/modelPolicyBot.js`
