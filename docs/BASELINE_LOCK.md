# Baseline Lock (v1)

이 문서는 모델 사이클마다 비교 기준을 고정하기 위한 운영 규칙이다.

## 1) 고정 기준

- Primary gate baseline: `Danmok V16`
- Anchor reference: `Heuristic V4`
- 평가 프로토콜: `1000 + 1000` (선후공 교차)
- 우선 지표: `avgGoldDelta` > `cumulativeGoldDelta` > `winRate` > `GO 실패율`

## 2) 승격 규칙

- Hard gate: `candidate vs V16` 2000판 집계 `avgGoldDelta > 0` 이어야 함.
- Soft gate: `candidate vs V4`는 즉시 승격 조건이 아니라, 이전 후보 대비 손실 축소를 모니터링.
- Safety gate: `GO 실패율` 급악화 시 보류.

## 3) 실행 스크립트

고정 실행 스크립트:

- `scripts/run_baseline_lock.ps1`

Dry-run (명령만 출력):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline_lock.ps1 `
  -CandidateTag v18 `
  -CandidateAttackModel models/policy-danmokv18-attack.json `
  -CandidateDefenseModel models/policy-danmokv18-defense.json `
  -CandidateValueModel models/value-danmokv18-gold.json
```

실행:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline_lock.ps1 `
  -CandidateTag v18 `
  -CandidateAttackModel models/policy-danmokv18-attack.json `
  -CandidateDefenseModel models/policy-danmokv18-defense.json `
  -CandidateValueModel models/value-danmokv18-gold.json `
  -Execute
```

## 4) 산출물

- 실행 계획 파일:
  - `logs/<candidateTag>_baseline_lock_plan.json`
- 매치 로그/리포트:
  - `logs/<candidateTag>_vs_v16_1000.jsonl`
  - `logs/v16_vs_<candidateTag>_1000.jsonl`
  - `logs/<candidateTag>_vs_v4_1000.jsonl`
  - `logs/v4_vs_<candidateTag>_1000.jsonl`
  - 각 파일별 `-report.json`

## 5) 참고 설정

- `configs/baseline_lock.json` 에 고정 기준/규칙이 정의되어 있다.

