# train_dual_net 설계 초안 (코드 전환 전 문서)

## 1) 목표
- 기존 `01_train_policy.py` + `02_train_value.py` 분리 학습을, 단일 PyTorch 모델로 통합한다.
- 당장 RL 전체 전환은 하지 않고, 현재 로그 기반 파이프라인에서 성능/속도 개선을 먼저 확인한다.

## 2) 모델 개요
- 파일명(예정): `scripts/train_dual_net.py`
- 구조: `Shared Trunk MLP + Policy Head + Value Head`
  - 입력: 상태/의사결정 feature 벡터
  - 공유층: 3~4개 FC 레이어 (ReLU, Dropout 선택)
  - Policy head: 행동 확률(logits)
  - Value head: 스칼라 기대값(최종 점수차/골드)

## 3) 입력/라벨
- 입력:
  - 기존 01/02 feature를 통합
  - 범주형: phase, order, decision_type, action/candidate context
  - 수치형: deck/hand/go count, immediate reward, 기타 파생 지표
  - 1차 버전은 현 해시 feature 재사용 가능 (이후 embedding 전환)
- 라벨:
  - Policy: 실제 선택 action index (cross-entropy)
  - Value: 최종 점수차 기반 스칼라 (MSE/Huber)

## 4) Loss
- `L = L_policy + lambda_value * L_value`
- 초기 권장:
  - `L_policy = CE(logits, action_target)`
  - `L_value = Huber(value_pred, value_target)`
  - `lambda_value = 0.5` (초기값, 튜닝 대상)

## 5) 학습/추론 운영
- 학습:
  - GPU 기본 (`cuda`)
  - mixed precision(torch.cuda.amp) 옵션
  - batch-size 우선 탐색: 4096 -> 8192 -> OOM 전까지
- 추론:
  - 정책 확률 + 가치 추정 동시 사용
  - 기존 bot 경로에 점진적으로 결합

## 6) 평가 기준 (전환 게이트)
- 보조 지표:
  - policy top1 / nll
  - value mae / rmse
- 공식 지표:
  - 챔피언전 실전 결정승률(20만판, report 기준)
- 전환 승인 조건:
  - 최소 2회 이상 반복 실험에서 기존 대비 결정승률 유의 개선

## 7) 단계별 도입 계획
1. `train_dual_net.py` 최소 버전 구현 (현 feature 재사용)
2. `04_run_pipeline.py`에 `--dual` 실험 경로 추가
3. 오프라인 + 실전 A/B 비교 자동화
4. 성능 개선이 확인되면 기본 파이프라인 승격

## 8) 범위 제외 (현재 단계)
- PPO/AlphaZero-lite 전면 도입
- MCTS 상시 추론
- 대규모 시퀀스 모델(Transformer) 전환

현재 문서는 설계 초안이며, 코드 전환은 실전 승률 개선 근거 확보 후 진행한다.
