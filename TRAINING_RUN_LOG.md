# TRAINING RUN LOG

## 목적
- 이 문서는 `scripts/` 폴더의 파일 역할만 기록한다.

## scripts 역할
- `scripts/selfplay_simulator.mjs`
  - AI vs AI self-play 본체.
  - `compact`, `delta`, `train` 로그 모드 지원.
  - 리포트(`*-report.json`) 생성.

- `scripts/run_parallel_selfplay.py`
  - self-play를 멀티 프로세스로 병렬 실행.
  - shard를 자동 병합해 하나의 JSONL/리포트로 생성.

- `scripts/run_league_selfplay.py`
  - 리그 설정(`league_config.example.json`) 기반 데이터 생성기.
  - 상대 풀 가중치 샘플링으로 self-play 데이터를 만든다.

- `scripts/01_train_policy.py`
  - 정책 모델 학습.
  - 빈도 기반 정책 JSON 출력.

- `scripts/02_train_value.py`
  - 가치 모델 학습.
  - 점수/골드 기대값 기반 회귀 모델 JSON 출력.
  - CPU/CUDA 옵션 지원.

- `scripts/03_evaluate.py`
  - old/new 정책-가치 모델 비교 평가.
  - 리포트 JSON 생성.

- `scripts/04_run_pipeline.py`
  - `01 -> 02 -> 03` 순차 실행 파이프라인.
  - `--fast` 프리셋 지원.

- `scripts/run-fast-loop.ps1`
  - 데이터 생성 + fast 파이프라인을 한 번에 실행.
  - 병렬 self-play/리그 self-play 둘 다 지원.

- `scripts/run_champion_cycle.ps1`
  - 챔피언/도전자 A-B 사이클 자동 실행.
  - 승격 판정, 요약 저장, Resume 지원.

- `scripts/update_champion_scoreboard.py`
  - 챔피언 사이클 summary를 모아 점수표(JSON/Markdown) 갱신.

- `scripts/train_dual_net.py`
  - Dual-head 학습기(EmbeddingBag + MLP).
  - `.pt` 캐시, AMP, early stopping, go/stop 가중치 지원.

- `scripts/TRAIN_DUAL_NET_DESIGN.md`
  - `train_dual_net.py` 설계/운영 메모.

- `scripts/league_config.example.json`
  - 리그 self-play 예시 설정 파일.

- `scripts/run-with-ram-guard.ps1`
  - 메모리 사용량 가드 실행 스크립트.

- `scripts/slice-month-webp-to-svg.mjs`
  - 이미지 자산 처리 유틸리티.
