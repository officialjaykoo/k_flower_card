# TRAINING RUN LOG

이 문서는 학습/셀프플레이 실행 방법, 산출 파일, 로그 필드 의미를 운영 기준으로 정리한 문서다.
향후 다른 스크립트/파이프라인 내용도 이 파일에 계속 추가한다.

## 0) 현재 운영 원칙 (요약)
- 시뮬레이션 엔진: `scripts/selfplay_simulator.mjs`
- 병렬 실행 래퍼: `scripts/run_parallel_selfplay.py`
- 리그 실행 래퍼: `scripts/run_league_selfplay.py`
- 기본 평가 기준: `averageGoldDeltaMySide` (report의 `primaryMetric`)
- 현재 코드 기준 train/delta는 파이프라인 상 단일화되어, 본문 로그 포맷은 동일하게 관리한다.

## 1) 실행 방법

### 1-1. 단일 실행 (Node)
```powershell
node scripts/selfplay_simulator.mjs 1000 logs/selfplay_1000.jsonl --policy-my-side=heuristic_v4 --policy-your-side=heuristic_v4
```

전체 옵션 목록 (`selfplay_simulator.mjs`):
- 입력 1: `games` (필수)
  - 기본(스위칭): **짝수 필수** (`mySide`/`yourSide` 선공 분배 50:50)
  - `--fixed-seats` 사용 시: 고정 모드(선공 스위칭 없음)
- 입력 2: `outPath` (선택, 생략 시 `logs/side-vs-side-<timestamp>.jsonl`)
- 입력 3: 양쪽 정책/모델
  - `--policy-my-side` (기본 `heuristic_v3`)
  - `--policy-your-side` (기본 `heuristic_v3`)
  - `--policy-model-my-side` (선택)
  - `--policy-model-your-side` (선택)
  - `--fixed-seats` (선택, 기본은 미지정=스위칭)

### 1-2. 병렬 실행 (Python)
```powershell
python scripts/run_parallel_selfplay.py 1000 --workers 4 --output logs/selfplay_parallel_1000.jsonl -- --policy-my-side=heuristic_v4 --policy-your-side=heuristic_v4
```

전체 옵션 목록 (`run_parallel_selfplay.py`):
- 입력 1: `games` (필수)
  - **짝수 필수** (내부적으로 worker shard도 짝수로 분배)
- 입력 2: 병렬/실행 제어
  - `--workers` (기본 `4`)
  - `--output` (선택, 생략 시 `logs/side-vs-side-parallel-<timestamp>.jsonl`)
  - `--node` (기본 `node`)
  - `--script` (기본 `scripts/selfplay_simulator.mjs`)
- 입력 3: 산출물 제어
  - `--shard-only` (선택): 병합 안 하고 shard만 유지 (merge JSONL 미생성)
  - `--keep-shards` (선택): 병합은 하되 worker shard 파일도 유지
  - `--emit-train-splits` (선택): shard별 학습용 `*.train_policy.jsonl` 추가 생성
- 입력 4: 시뮬레이터 전달 인자
  - `--` 뒤 인자는 그대로 `selfplay_simulator.mjs`로 전달
  - 예: `-- --policy-my-side=heuristic_v4 --policy-your-side=heuristic_v4 --fixed-seats`

### 1-3. 리그 실행 (Python)
```powershell
python scripts/run_league_selfplay.py --config scripts/league_config.json --total-games 1000 --workers 4 --output logs/league_1000.jsonl
```

메모:
- focal vs opponents를 가중치로 매칭.
- chunk 단위로 병렬 실행 후 최종 merge.

## 2) 주요 산출 파일

### 2-1. selfplay_simulator 직접 실행 시
- 메인 로그: `<out>.jsonl`
- 리포트: `<out>-report.json`
- 카탈로그: `logs/catalog/cards-catalog.json`

### 2-2. run_parallel_selfplay 실행 시
- 병합 로그: `--output` 경로의 `.jsonl`
- 병합 리포트: 동일 경로의 `-report.json`
- shard 로그(옵션): `<base>.partN.jsonl`
- shard 리포트(옵션): `<base>.partN-report.json`
- train split(옵션): `<base>.partN.train_policy.jsonl`

### 2-3. run_league_selfplay 실행 시
- 병합 로그: `--output` 경로의 `.jsonl`
- 리그 리포트: 동일 경로의 `-report.json`
- chunk 임시 디렉터리: `<base>.chunks-<timestamp>/` (옵션에 따라 자동 정리)

## 3) JSONL 한 줄(게임 1개) 필드와 역할

아래는 `selfplay_simulator.mjs`의 기본 persisted line 구조다.

실제 저장되는 필드는 아래와 같다.

```json
{
  "ver": {
    "s": 2,
    "f": 14,
    "t": 2
  },
  "game": 1,
  "runId": "run-2026-...-a1b2c3",
  "seatMode": "switching | fixed",
  "seed": "sim-...",
  "steps": 123,
  "firstAttackerActor": "A",
  "policy": {
    "mySide": "heuristic_v4 | model:<file>",
    "yourSide": "heuristic_v4 | model:<file>"
  },
  "winner": "mySide | yourSide | draw | unknown",
  "score": {
    "mySide": 0,
    "yourSide": 0
  },
  "initialGoldMy": 0,
  "initialGoldYour": 0,
  "finalGoldMy": 0,
  "finalGoldYour": 0,
  "learningRole": "ATTACK | DEFENSE | NEUTRAL",
  "decision_trace": [
    {
      "seq": 71,
      "t": 0,
      "a": "mySide | yourSide",
      "dt": "play | match | option",
      "cc": 0,
      "la": ["legal action 1", "legal action 2"],
      "ch": "chosen compact value",
      "ir": 0.0,
      "dc": {
        "p": 0,
        "d": 0,
        "hs": 0,
        "hd": 0,
        "gs": 0,
        "go": 0,
        "rp": 0,
        "rg": 0,
        "rm": 0,
        "ss": 0,
        "so": 0,
        "jps": 0,
        "jpo": 0,
        "jas": 0,
        "jao": 0,
        "sjt": 0,
        "sjo": 0,
        "sgt": 0,
        "ojt": 0,
        "ojo": 0,
        "ogt": 0,
        "gsd": 0,
        "sm": 0,
        "sy": 0,
        "sd": 0
      },
      "keepBy": [
        { "kind": "self", "trigger": "goStopOption" },
        { "kind": "from", "fromTurn": 37, "trigger": "riskShiftUp" }
      ]
    }
  ]
}
```

`dc` 축약 키 맵 (예시와 동일 레벨/동일 순서):

```json
{
  "p": "phase",
  "d": "deckCount",
  "hs": "handCountSelf",
  "hd": "handCountDiff",
  "gs": "goCountSelf",
  "go": "goCountOpp",
  "rp": "piBakRisk",
  "rg": "gwangBakRisk",
  "rm": "mongBakRisk",
  "ss": "shakeCountSelf",
  "so": "shakeCountOpp",
  "jps": "jokboProgressSelfSum",
  "jpo": "jokboProgressOppSum",
  "jas": "jokboOneAwaySelfCount",
  "jao": "jokboOneAwayOppCount",
  "sjt": "selfJokboThreatProb",
  "sjo": "selfJokboOneAwayProb",
  "sgt": "selfGwangThreatProb",
  "ojt": "oppJokboThreatProb",
  "ojo": "oppJokboOneAwayProb",
  "ogt": "oppGwangThreatProb",
  "gsd": "goStopDeltaProxy",
  "sm": "scoreMySideAtTurn",
  "sy": "scoreYourSideAtTurn",
  "sd": "scoreDiff(actor 기준)"
}
```

`decision_trace` 핵심 규칙:
- `la`: 해당 시점 합법 액션 목록(학습 입력의 단일 기준)
- `ch`: 실제 선택 액션
- 정책 학습(01)은 `la`만 후보로 사용하며, `ch in la`가 아니면 샘플을 버림

`dc` 키 한글 설명(옆 설명):

| 키 | 원본 의미 | 한글 설명 |
|---|---|---|
| `p` | `phase` | 현재 의사결정 단계 |
| `d` | `deckCount` | 남은 덱 장수 |
| `hs` | `handCountSelf` | 행동 주체 손패 수 |
| `hd` | `handCountDiff` | 손패 수 차이(나-상대) |
| `gs` | `goCountSelf` | 행동 주체 GO 횟수 |
| `go` | `goCountOpp` | 상대 GO 횟수 |
| `rp` | `piBakRisk` | 피박 위험 여부(0/1) |
| `rg` | `gwangBakRisk` | 광박 위험 여부(0/1) |
| `rm` | `mongBakRisk` | 멍박 위험 여부(0/1) |
| `ss` | `shakeCountSelf` | 행동 주체 흔들기 누적 횟수 |
| `so` | `shakeCountOpp` | 상대 흔들기 누적 횟수 |
| `jps` | `jokboProgressSelfSum` | 내 족보 진행 총합 |
| `jpo` | `jokboProgressOppSum` | 상대 족보 진행 총합 |
| `jas` | `jokboOneAwaySelfCount` | 내 원어웨이 족보 개수 |
| `jao` | `jokboOneAwayOppCount` | 상대 원어웨이 족보 개수 |
| `sjt` | `selfJokboThreatProb` | 내 족보 위협 확률 |
| `sjo` | `selfJokboOneAwayProb` | 내 원어웨이 확률 |
| `sgt` | `selfGwangThreatProb` | 내 광 위협 확률 |
| `ojt` | `oppJokboThreatProb` | 상대 족보 위협 확률 |
| `ojo` | `oppJokboOneAwayProb` | 상대 원어웨이 확률 |
| `ogt` | `oppGwangThreatProb` | 상대 광 위협 확률 |
| `gsd` | `goStopDeltaProxy` | GO/STOP 판단 프록시 값 |
| `sm` | `scoreMySideAtTurn` | 해당 턴의 mySide 점수 |
| `sy` | `scoreYourSideAtTurn` | 해당 턴의 yourSide 점수 |
| `sd` | `scoreDiff` | 행동 주체 기준 점수차 |

추가 메모:
- `runId`는 실행 단위 식별자이며, 서로 다른 실행 로그를 병합해도 원천 run을 구분할 수 있다.
- `seatMode`는 해당 라인의 좌석 운영 모드(`switching`/`fixed`)를 의미한다.
- `decision_trace[].keepBy`는 보존 사유 배열:
  - `{ "kind":"self", "trigger":"<triggerName>" }`: 해당 턴 자체가 트리거
  - `{ "kind":"from", "fromTurn":<anchorTurn>, "trigger":"<triggerName>" }`: 다른 턴 트리거 범위(back/forward)에 포함되어 보존
  - `{ "kind":"context", "fromTurn":<anchorTurn> }`: `contextRadius` 확장으로 보존
  - `trigger` 값은 전체 이름 사용:
    - `earlyTurnForced`, `goStopOption`, `shakingYesOption`, `optionTurnOther`
    - `deckEmpty`, `specialEventPpuk`, `specialEventCore`
    - `bombDeclare`, `bombCountShift`
    - `riskShiftUp`, `riskShiftDown`, `riskShiftMixed`
    - `comboThreatEnter`, `terminalContext`
- `goldDeltaMy`, `bankrupt`는 제거됨(필요 시 `initial/finalGold*`로 계산 가능).
- `terminal` 필드는 제거됨(필요 시 `score`/`initialGold*`/`finalGold*` 기반으로 후처리 계산).
- `learningRole` 규칙은 현재 고정:
  - `ATTACK`: `winner == mySide && myScore >= 20`
  - `DEFENSE`: `winner == yourSide && oppScore <= 10`
  - 나머지 `NEUTRAL`

### 3-1) 결정 트레이스 저장 조건 (`decision_trace`)
- 소스:
  - 기본은 `kibo`의 `turn_end` 턴에서 생성
  - 추가로 option 턴(`optionEvents`)도 별도 레코드 생성
- 하드 필터:
  - 후보 수 `candidateCount < 2`는 저장하지 않음 (선택지 2개 이상만 저장)
- 현재 고정 설정:
  - `myTurnOnly=false` (양쪽 사이드 모두 기록)
  - `importantOnly=true` (중요턴 중심 저장)
  - `contextRadius=0`
  - `goStopPlus2=false`
- 중요턴 트리거(보존 범위 `back/forward`):
  - `earlyTurnForced`: `0 / 0`
  - `goStopOption`: 기본 `1 / 1` (예외: `choose_go` + `goStopPlus2=true`면 `1 / 2`)
  - `shakingYesOption`: `0 / 1`
  - `optionTurnOther`: `0 / 1`
  - `deckEmpty`: `1 / 0`
  - `specialEventPpuk`: `1 / 1`
  - `specialEventCore`: `1 / 0`
  - `bombDeclare`: `1 / 1`
  - `bombCountShift`: `0 / 1`
  - `riskShiftUp`: `1 / 0`
  - `riskShiftDown`: `0 / 1`
  - `riskShiftMixed`: `0 / 0`
  - `comboThreatEnter`: `0 / 0`
  - `terminalContext`: `0 / 0` (적용 위치: `records.length - 2`, 종료 직전 문맥 1개 전 레코드)
- 최종 보존 규칙:
  - 트리거가 있으면 트리거별 `back/forward` 범위만 보존
  - 여러 트리거가 겹치면 합집합으로 보존
  - `contextRadius > 0`이면 최종 keep 인덱스를 반경만큼 추가 확장
  - 저장된 각 레코드에는 보존 사유가 `keepBy[]`로 함께 기록됨
  - 트리거가 하나도 없으면(예외) 하드 필터를 통과한 레코드 전체를 보존

### 3-2) 순서 추적 필드
- `decision_trace[].seq`:
  - 같은 `t`에서 여러 레코드가 생길 수 있을 때(예: option/이벤트) 순서를 구분하는 1차 키
  - 정렬 권장: `seq` 오름차순, 동률이면 `t` 오름차순

## 4) report JSON 필드와 역할

### 4-1. 공통
- `logMode`, `games`, `completed`, `winners`
- `learningRole.attackCount/defenseCount/neutralCount`
- `learningRole.attackScoreMin`, `learningRole.defenseOppScoreMax`
- `sideStats`:
  - `mySideWinRate`, `yourSideWinRate`, `drawRate`
  - `averageScoreMySide`, `averageScoreYourSide`
- `economy`:
  - `averageGoldMySide`, `averageGoldYourSide`
  - `averageGoldDeltaMySide`
  - `cumulativeGoldDeltaOver1000`
  - `cumulativeGoldDeltaMySideFirst1000`
- `bankrupt`:
  - inflicted/suffered/diff/rate, resets
- `primaryMetric`: `averageGoldDeltaMySide`

### 4-2. full 확장
- `catalogPath`
- `nagariRate`
- `eventFrequencyPerGame`
- `goStopEfficiencyAvg`
- `goDecision` (declared/success/successRate)
- `stealEfficiency`
- `bakEscapeRate`, `bakBreakdown`
- `luckSkillIndex`

### 4-3. run_parallel_selfplay 리포트 확장
- `logMode: merged_parallel`
- `workers`, `shards`
- `trainSplits.policyShards` (옵션 시)

### 4-4. run_league_selfplay 리포트 확장
- `logMode: league_parallel`
- `created_at`, `config`
- `chunks[]` (chunk별 mySide/yourSide, games, output, report)
- `matchups`

## 5) 실무 체크리스트
- 실행 전:
  - 출력 경로(`logs/...`) 충돌 여부 확인
  - 모델 경로 오탈자 확인 (`models/...json`)
- 실행 후:
  - `<out>-report.json`의 `games`, `completed` 일치 확인
  - `averageGoldDeltaMySide` 우선 확인
  - `learningRole` 분포(ATTACK/DEFENSE/NEUTRAL) 확인
- 장기 운영:
  - baseline 비교 시 게임 수/옵션 고정
  - 리포트 원본 파일 경로를 문서에 반드시 기록

## 6) 다음 업데이트 예정
- 정책 학습(01) 입력 사양
- 가치 학습(02) 입력 사양
- 모델 버전별 실험 이력 템플릿
