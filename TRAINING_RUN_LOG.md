# TRAINING RUN LOG

## 2026-02-20 시뮬레이션/모델링 업데이트 (scripts 기준 재정리)

- 기준 파일:
  - `scripts/selfplay_simulator.mjs`
  - `scripts/run_parallel_selfplay.py`
  - `scripts/01_train_policy.py`
  - `scripts/02_train_value.py`
  - `scripts/03_evaluate.py`
  - `scripts/build_errata_dataset.py`
  - `scripts/run_league_selfplay.py`
  - `scripts/run_champion_cycle.ps1`

### A. selfplay 생성기(`selfplay_simulator.mjs`) 핵심 변경점

- 로그 스키마 버전 고정 필드 추가:
  - `ver.s=2`, `ver.f=14`, `ver.t=2`
- 실행 모드:
  - `--log-mode=train|delta`만 허용
  - 학습 모드(`train/delta`)에서 lean kibo 사용
- 모델 기반 의사결정 경로 확장:
  - 사이드별 `--policy-model-*`, `--value-model-*` 지원
  - MoE 분기(`--policy-model-attack-*`, `--policy-model-defense-*`) 지원
  - `--model-only` 사용 시 모델 추론 실패를 즉시 에러 처리
- 탐색(Exploration) 파라미터 추가/강화:
  - `--train-explore-rate`
  - `--train-explore-rate-comeback`
  - `--train-explore-decay-k`
  - `--train-explore-min`
  - `--train-explore-comeback-min`
- `decision_trace` 압축/호환 필드:
  - `ch`(compact chosen), `ck`(context key), `tm`(trigger bitmask) 사용
  - 기존 `c/s/at`, `tg`와 병행 호환
- 트리거 비트셋 정식화(10개):
  - `earlyTurnForced`, `goStopOption`, `shakingYesOption`, `optionTurnOther`, `deckEmpty`, `specialEvent`, `bombEvent`, `riskShift`, `comboThreatEnter`, `terminalContext`
- 경제 루프 반영:
  - 게임 간 골드 carry-over
  - 한쪽 파산 시 세션 골드 리셋

### B. 병렬 생성/리포트(`run_parallel_selfplay.py`) 변경점

- 총 게임 수는 짝수 강제(워커 분배도 짝수 단위)
- `--emit-train-splits` 추가:
  - shard별 최소 학습 로그(`*.train_policy.jsonl`) 생성
  - 최소 trace 필드: `a,o,dt,cc,ch,ck`
- `--shard-only` 추가:
  - 병합 JSONL 생략, shard 산출물 중심 운영
- 병합 리포트 지표 확장:
  - `economy.averageGoldDeltaMySide`
  - `economy.cumulativeGoldDeltaOver1000`
  - `economy.cumulativeGoldDeltaMySideFirst1000`
  - `bankrupt.*`(inflicted/suffered/rate/reset)

### C. 정책 학습(`01_train_policy.py`) 변경점

- 정책 필터 규칙 파일 도입:
  - 기본 `configs/policy_filter_rules.json`
  - 기본 규칙 `min_candidate_count=2`
- 컨텍스트 키 생성 시 `ck` 우선 사용
- 샘플 캐시 체계:
  - `--cache-backend jsonl|lmdb`
  - 입력 매니페스트/설정 기반 cache 재사용 판정
  - `--skip-train-metrics` 지원

### D. 가치 학습(`02_train_value.py`) 변경점

- 학습 샘플 단위:
  - 1게임 1샘플이 아니라 `decision_trace`의 각 턴이 1샘플
  - 기본 필터는 `cc >= 2`, 예외(`declare_bomb`/특정 트리거)는 보존
- 타깃(`y`) 설계:
  - 기본 `--target-mode gold`에서 actor 기준 `gold delta` 회귀
  - `goldDelta*` 우선, 없으면 `finalGold - initialGold`로 보완
  - 대안 모드 `--target-mode score` 지원
- 피처 설계:
  - 토큰: `phase/order/decision_type/action/deck_bucket/...` 해시 인코딩
  - 수치: `deck_count`, `hand_self`, `hand_opp`, `score_diff`, `bak_risk_total`,
    `jokbo*`, `opp*ThreatProb`, `go_stop_delta_proxy`, `immediate_reward`
  - 신규 로그 호환:
    - `scoreDiff` 우선 사용, 구버전은 `currentScoreSelf-currentScoreOpp` fallback
    - `handCountOpp` 없으면 `handCountSelf-handCountDiff`로 복원
- 학습 방식:
  - 선형 해시 회귀(`value_linear_hash_v1`) + MSE
  - 옵티마이저: SGD, L2는 `weight_decay`로 적용
  - 검증 분할: deterministic hash split(`--valid-ratio`)
  - 지표: train/valid `MSE, RMSE, MAE, Pearson`
- 실행/백엔드:
  - `--device cpu|cuda` 지원(현재 기본 `cuda`)
  - sparse sample cache(`jsonl/lmdb`) + `shuffle-buffer`
  - 이번 실행 파라미터:
    - `--device cuda --epochs 8 --lr 0.008 --l2 3e-6 --cache-backend jsonl --sample-cache none`

### E. 평가/승급 게이트(`03_evaluate.py`) 변경점

- North-star 중심 리포트 구조:
  - `avg_gold_delta_my` + 95% CI
  - `go_fail_rate_total` / side별 go fail
- 승급 추천 게이트(`promotion_recommended`):
  - `gold_significant_positive`
  - `go_fail_rate_safe`(총 go fail rate <= 0.08)
  - `value_not_worse`

### F. 에라타 데이터셋(`build_errata_dataset.py`) 변경점

- 정책/가치 에러 임계치 자동 산정:
  - 정책: `--policy-loss-quantile`
  - 가치: `--value-error-min` + 선택 `--value-error-quantile`
- 산출물 분리:
  - 기본 errata JSONL + oversampled JSONL + summary JSON

### G. 리그/챔피언 운영 스크립트 변경점

- `run_league_selfplay.py`:
  - 가중치 상대 풀 기반 매치업 샘플링
  - chunk 진행률 기반 explore decay 스케줄
  - `model_only_when_possible` 시 자동 `--model-only` 전달
- `run_champion_cycle.ps1`:
  - 승급 판단에 `ZScoreGate`, `GoFailRateTolerance`, `WorstSingleLoss` 비교 반영
  - 라운드 상태 파일(`logs/champ-cycle-state-*.json`) 기반 resume 지원

### H. 운영 주의

- 본 업데이트는 `scripts` 정적 확인 기준 반영이며, 시뮬레이션 실행은 수행하지 않음.
- 저장 인코딩은 UTF-8 BOM 유지.
## 1) `decision_trace` 기록 시점(핵심)

`decision_trace`는 턴마다 무조건 저장하지 않고, 아래 기준을 통과한 턴만 기록한다.

- 수집 기준:
  - `kibo.type === "turn_end"`에서 턴 단위 레코드 생성
  - `selectionPool.decisionType === "option"`인 별도 option 이벤트도 추가 레코드 생성
- 턴 확정:
  - 루프에서 `nextTurnSeq > prevTurnSeq`일 때 해당 턴의 before/after 컨텍스트를 매핑
- 사이드 필터:
  - 기본: 양쪽 턴 기록
  - `--trace-my-turn-only`: `mySide`만 기록
- 후보 수 필터:
  - `train`: `candidateCount >= 1`
  - `delta`: `candidateCount >= 2`
- 중요턴 필터:
  - 기본 ON(`--trace-important-only`)
  - OFF(`--trace-all-candidate-turns`) 시 후보 수 조건만 통과하면 기록
- 컨텍스트 확장:
  - `--trace-context-radius`로 중요턴 주변 반경 보존
  - `--trace-go-stop-plus2`로 `goStopOption(go 선택)`의 forward 범위를 `+2`까지 확장

## 2) 중요턴 10개 트리거 상세

아래 10개 그룹으로 관리한다.  
내부 raw 트리거명(`specialEventPpuk`, `specialEventCore` 등)은 비트마스크(`tm`)로 10개 그룹에 매핑된다.

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
- raw 매핑:
  - `specialEventPpuk` -> `PPUK` (`T-1, T, T+1`)
  - `specialEventCore` -> `JJOB/DDADAK/PANSSEUL` 등 (`T-1, T`)

7. `bombEvent`
- 조건: 폭탄 선언 또는 폭탄 카운트 변화
- raw 매핑:
  - `bombDeclare` (`declare_bomb`): `T-1, T, T+1`
  - `bombCountShift`(카운트 변화): `T, T+1`

8. `riskShift`
- 조건: `piBakRisk/gwangBakRisk/mongBakRisk` 변화
- raw 매핑:
  - `riskShiftUp`(증가): `T-1, T`
  - `riskShiftDown`(감소): `T, T+1`
  - `riskShiftMixed`(혼합): `T`

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

`train` 모드 레코드(현재 구현):

- `t`: 턴 번호
- `a`: 액터 사이드(`mySide`/`yourSide`)
- `dt`: 의사결정 타입(`play`/`match`/`option`)
- `cc`: 후보 수
- `ch`: compact 선택값(카드 id / 옵션 alias)
- `ir`: 중간보상(immediate reward)
- `dc`: 압축 decision context
- `tm`: 트리거 비트마스크(해당 시)

참고:

- `tm`이 없으면 트리거 없는 일반 턴 레코드
- `option` 이벤트는 별도 경로로 생성되지만 동일 필드 스키마로 기록됨
- 과거/보조 스키마(`o`, `ck`, `at`, `c`, `s`, `tg`)는 현재 `train` 기록 기본 필드가 아님

`delta` 모드 레코드(현재 구현):

- `t`, `a`, `ch`
- `delta`:
  - `deck`, `handSelf`, `handOpp`
  - `piSteal`, `goldSteal`
  - `capHand`, `capFlip`, `events`
- `reasoning`:
  - `policy`, `candidatesCount`, `evaluation`

`dc` 필드:

- 현재 로그 저장 기준(`compactDecisionContextForTrace`)은 17개:
- `phase`
- `deckCount`
- `handCountSelf`
- `handCountDiff`
- `goCountSelf`
- `goCountOpp`
- `piBakRisk`
- `gwangBakRisk`
- `mongBakRisk`
- `jokboProgressOppSum`
- `jokboOneAwaySelfCount`
- `jokboOneAwayOppCount`
- `oppJokboThreatProb`
- `oppJokboOneAwayProb`
- `oppGwangThreatProb`
- `goStopDeltaProxy`
- `scoreDiff`

`dc` 축소 검토안(요청 반영):

- 제거(적용):
  - `carryOverMultiplier`
- 축약(적용):
  - `handCountSelf`, `handCountOpp` -> `handCountSelf` + `handCountDiff`
- 통합(적용):
  - `currentScoreSelf`, `currentScoreOpp` -> `scoreDiff`
- 단계적 검토:
  - 현재 없음

## 4) 게임 루트(1게임 1레코드)

- `ver`(schema/feature/trigger dict 버전)
- `run`(각 side 에이전트 라벨)
- `game`, `seed`, `steps`
- `firstAttackerActor`
- `policy`(각 side 라벨)
- `winner`, `score`
- `initialGoldMy`, `initialGoldYour`
- `finalGoldMy`, `finalGoldYour`
- `goldDeltaMy`
- `bankrupt`(`mySide`, `yourSide`)
- `decision_trace`

참고:

- `logMode`는 라인 필드로 저장하지 않고, 실행 인자/리포트(`*-report.json`)에서 관리
- `goldDeltaYour`, `gold`, `sessionReset` 등은 현재 기본 라인 스키마에서 제거됨

## 5) 관련 파일

- `scripts/selfplay_simulator.mjs`
- `scripts/run_parallel_selfplay.py`
- `scripts/01_train_policy.py`
- `scripts/02_train_value.py`
- `scripts/03_evaluate.py`
- `scripts/build_errata_dataset.py`
- `scripts/run_league_selfplay.py`
- `scripts/run_champion_cycle.ps1`
