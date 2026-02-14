## 현재 상태 (챔피언/도전자)
- Champion: `센돌 (sendol)`
- Challenger: `돌바람 (dolbaram)`
- 판정 기준: Challenger 결정승률(`draw 제외`) `>= 52%`이면 챔피언 교체
- Champion policy: `models/policy-sendol.json` (`policy-randomplus-next-200k` 기반)
- Champion value: `models/value-sendol.json` (`value-randomplus` 유지)
- Challenger policy: `models/policy-dolbaram.json` (현재 `v3`)
- Challenger value: `models/value-dolbaram.json` (현재 `v3`)

## 모델 소개
- 센돌 (Champion):
- AI vs AI 거의 랜덤 self-play 누적 약 300만판 규모 기반 모델에서, 정책만 `next`로 갱신한 챔피언 모델
- policy: `models/policy-sendol.json`
- value: `models/value-sendol.json`

- 돌바람 (Challenger):
- 최신 대결 데이터(랜덤_v2 vs 휴리스틱_v1 진영교차)로 학습된 도전자 모델
- policy: `models/policy-dolbaram.json`
- value: `models/value-dolbaram.json`

## 돌바람 버전 규칙
- 돌바람은 챔피언(센돌)에게 질 때마다 `정책(Policy)`만 업데이트하여 버전을 올림
- 가치(Value)는 평가 열세 시 기존 유지
- 현재 버전: `돌바람 v3`
- 버전 파일:
- `models/policy-dolbaram-v3.json`
- `models/value-dolbaram-v3.json`

## 모델 전용 대결 모드 (fallback 없음)
- 실행 옵션: `--model-only`
- 의미: 휴리스틱/랜덤 정책 fallback 금지, 모델이 직접 선택하지 못하면 즉시 에러 종료
- 예시: 
- `node scripts/simulate-ai-vs-ai.mjs 100000 logs/match-randomplus-vs-dolbaram-a.jsonl --log-mode=train --model-only --policy-model-a=models/policy-randomplus.json --value-model-a=models/value-randomplus.json --policy-model-b=models/policy-dolbaram.json --value-model-b=models/value-dolbaram.json`
- `node scripts/simulate-ai-vs-ai.mjs 100000 logs/match-randomplus-vs-dolbaram-b.jsonl --log-mode=train --model-only --policy-model-a=models/policy-dolbaram.json --value-model-a=models/value-dolbaram.json --policy-model-b=models/policy-randomplus.json --value-model-b=models/value-randomplus.json`

## 단계명
- 00 = 데이터 생성 (AI vs AI self-play를 병렬)
- 01 = 정책 모델 (Policy)
- 02 = 가치 모델 (Value)
- 03 = 평가 (Evaluate)
- 04 = 실전 판정(챔피언전)
- 05 = 점수표 누적

## 운영 규칙 (고정)
- 현재 대진: `센돌(Champion)` vs `돌바람(Challenger)`
- 챔피언전은 20만판(모델 전용, `--model-only`)으로 판정
- 챔피언 교체 조건: 도전자 결정승률(`draw 제외`) `>= 52%`

### 패자 개선 절차 (사용자 합의)
1. 챔피언전에서 진 쪽(현재: 돌바람)만 개선 대상으로 지정
2. 최근 챔피언전 20만판 로그를 학습 데이터로 사용
3. `01`로 패자 `next 정책` 생성
4. `02`로 패자 `next 가치` 생성
5. `03`에서 비교 대상은 **패자 old vs 패자 next** (챔피언과 직접 비교 아님)
6. 패자 내부 비교에서 통과한 모델만 챔피언에게 재도전
7. 재도전 승률이 52% 미만이면 챔피언 유지, 패자 개선 루프 반복

## 이 PC 안전 운용 한계 (실측 기준)
- 기준 장비: 데스크탑 (`i7-6700K`, RAM 16GB, SSD)
- 시뮬레이션:
- `100,000판` 단위 안정 운용 가능
- 정책(01):
- `200,000판` 입력은 안정적으로 완료
- 가치(02):
- 안전 구간: `200,000판 + --dim 128 + --epochs 1`
- 위험 구간: `300,000판+` 또는 `--dim 384/512` (RAM 95% 근접 가능)
- 권장 기본 프로파일:
- `20만판 루프`, `value dim=128`, `epochs=1`
- 확장 시에는 `dim` 또는 `epochs`를 한 번에 하나씩만 증가

## 실행 시간 측정 (PowerShell)
01. 정책 모델 (10만+10만)
- Measure-Command { py -3 scripts/01_train_policy.py --input logs/match-randomplus-vs-dolbaram-a.jsonl logs/match-randomplus-vs-dolbaram-b.jsonl --output models/policy-randomplus-next-200k.json }
- 64.9초
02. 가치 모델 (10만+10만 --dim 128 -- epochs 1)
- Measure-Command { py -3 scripts/02_train_value.py --input logs/match-randomplus-vs-dolbaram-a.jsonl logs/match-randomplus-vs-dolbaram-b.jsonl --output models/value-randomplus-next-200k-d128-e1.json --dim 128 --epochs 1 }
- 3분 3초 (183.1초)
- 디바이스 옵션: `--device cuda|cpu` (기본값 `cuda`)
03. 평가 모델 (20만)
- Measure-Command { py -3 scripts/03_evaluate.py --input logs/match-randomplus-vs-dolbaram-a.jsonl logs/match-randomplus-vs-dolbaram-b.jsonl --policy-old models/policy-randomplus.json --policy-new models/policy-randomplus-next-200k.json --value-old models/value-randomplus.json --value-new models/value-randomplus-next-200k-d128-e1.json --output logs/model-eval-randomplus-next-vs-randomplus.json }
-  4분 42.3(282.3초)

00. 모델전용 챔피언전 (센돌 vs 돌바람 v2, 20만)
- Measure-Command { node scripts/simulate-ai-vs-ai.mjs 200000 logs/match-sendol-vs-dolbaram-v2-200k.jsonl --log-mode=train --model-only --policy-model-a=models/policy-sendol.json --value-model-a=models/value-sendol.json --policy-model-b=models/policy-dolbaram.json --value-model-b=models/value-dolbaram.json }
- 11분 50.0초 (710.0초)

## 빠른 운영 명령 (최신)
- 병렬 self-play 20만(워커 4):
- `py -3 scripts/parallel_simulate_ai_vs_ai.py 200000 --workers 4 --output logs/train-parallel-200k.jsonl -- --log-mode=train --policy-human=heuristic_v1 --policy-ai=heuristic_v2`
- 정책 빠른 학습(샘플 제한 + 메트릭 생략):
- `py -3 scripts/01_train_policy.py --input logs/train-parallel-200k.jsonl --output models/policy-fast.json --max-samples 100000 --skip-train-metrics`
- 가치 빠른 학습(GPU 기본 튜닝):
- `py -3 scripts/02_train_value.py --input logs/train-parallel-200k.jsonl --output models/value-fast.json --device cuda --dim 128 --epochs 1 --batch-size 8192 --progress-every 0 --max-samples 200000`
- 파이프라인 fast 프리셋:
- `py -3 scripts/04_run_pipeline.py --input logs/train-parallel-200k.jsonl --tag fast-v1 --fast`

## 모델 성장 점수표 (리포트 파일 기준)
- 기준:
- 실전 점수 = `결정승률(%)` (50이 동률, 50 초과면 우세)
- 오프라인 점수 = `model-eval policy.duel.new_win_rate(%)`

### 실전(챔피언전) 점수
- 출처: `logs/match-sendol-vs-dolbaram-200k-report.json`
- 돌바람 v2 점수: `49.6081`
- 센돌 점수: `50.3919`
- 출처: `logs/match-sendol-vs-dolbaram-v2-200k-report.json`
- 돌바람 v3 점수: `49.7734`
- 센돌 점수: `50.2266`
- 성장폭(돌바람): `+0.1653`p (`49.6081 -> 49.7734`)

- rp-hv1: 65.0873
- stage4-rv2-hv1: 50.6818
- randomplus-next: 82.2608
- dolbaram-v3-vs-v2: 55.9059

### 오프라인(model-eval) 점수
- 출처: `logs/model-eval-rp-hv1-200k-vs-randombest.json`
- new 점수: `65.0873`
- 출처: `logs/model-eval-stage4-rv2-hv1-vs-current.json`
- new 점수: `50.6818`
- 출처: `logs/model-eval-randomplus-next-vs-randomplus.json`
- new 점수: `82.2608`
- 출처: `logs/model-eval-dolbaram-v3-vs-v2.json`
- new 점수: `55.9059`
- 성장폭(v3 vs v2): `+5.9059`p (오프라인 비교 기준)

## 방향성 합의 (2026-02-14)
- 결론:
- 당장은 현재 운영 방식(시뮬레이션 -> 01 정책 -> 02 가치 -> 03 평가)을 유지한다.
- 이유:
- 지금 즉시 아키텍처 전면 전환(통합 신경망 + RL)보다, 속도/운영 안정화와 평가 일관성 확보가 우선이다.
- 해석:
- "지금은 기존 방식 유지, 대신 결과가 더 빨리 나오도록 파이프라인을 최적화한다."
- 중장기:
- 통합 PyTorch 모델(train_dual_net)과 self-play 강화학습은 다음 단계로 준비한다.

## 실행 로드맵 (우선순위)
1. 속도 최적화 고정 운영 (즉시)
- 병렬 self-play 기본화: `scripts/parallel_simulate_ai_vs_ai.py` 사용
- 01 빠른 옵션 기본화: `--max-samples`, `--skip-train-metrics`
- 02 GPU 튜닝 기본화: `--device cuda --batch-size 8192 --progress-every 0`
- fast 루프 기본 명령: `py -3 scripts/04_run_pipeline.py --input <train_logs> --tag <tag> --fast`

2. 평가 기준 고정 (즉시)
- 공식 점수는 챔피언전 실전 결정승률(20만판, model-only)로 고정
- `model-eval`은 보조 지표로 사용
- 모든 결과는 리포트 파일 기준으로 본 문서에 누적

3. 운영 자동화 강화 (단기)
- 챔피언/도전자 사이클 스크립트 표준화
- 실패/중단 복구 지점(로그 shard, 모델 체크포인트) 명시
- 실행 시간/자원 사용량(RAM/GPU) 측정값을 본 문서에 누적

4. 모델 전환 준비 (중기)
- `train_dual_net.py` 설계안 문서화 (입력/라벨/loss/평가)
- 기존 01/02와 병행 비교 가능한 형태로 최소 버전 구현
- 전환 조건: 실전 결정승률 개선이 통계적으로 반복 확인될 때

5. 차세대 방법 도입 (중장기)
- self-play 학습 루프 고도화
- 필요 시 PPO/AlphaZero-lite/MCTS를 단계적으로 도입
- 도입 기준: 구현 복잡도 대비 실전 승률 개선이 명확할 때만 진행

## 고정 운영 모드 (즉시 적용)
- Fast 학습 루프(기본):
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-fast-loop.ps1 -TrainGames 200000 -Workers 4 -Tag fast`
- Fast 학습 루프(센돌 vs 돌바람 모델 데이터 생성):
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-fast-loop.ps1 -TrainGames 200000 -Workers 4 -Tag sendol-vs-dolbaram -PolicyModelHuman models/policy-sendol.json -ValueModelHuman models/value-sendol.json -PolicyModelAi models/policy-dolbaram.json -ValueModelAi models/value-dolbaram.json -ModelOnly`
- 챔피언 사이클(병렬 기본):
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-champion-cycle.ps1 -Champion sendol -Challenger dolbaram -GamesPerSide 100000 -Rounds 1 -Tag champ`
- 챔피언 사이클 복구:
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-champion-cycle.ps1 -Tag champ -Resume`
- 점수표 자동 누적(리포트/summary 기반):
- `py -3 scripts/update_champion_scoreboard.py`

## train_dual_net 준비 상태
- 설계 문서: `scripts/TRAIN_DUAL_NET_DESIGN.md`
- 현재 상태: 설계 완료(코드 전환 보류)
- 전환 조건: 챔피언전 실전 결정승률 개선이 반복 확인될 때

## 챔피언전 누적 점수표 (자동 생성)
- updated_at: 2026-02-14T13:51:42.056679+00:00
- source: logs/champ-cycle-*-summary.json (report 기반 산출)

| tag | challenger_before | champion_before | challenger_dec_win_rate | promoted | champion_after |
|---|---|---|---:|---:|---|

