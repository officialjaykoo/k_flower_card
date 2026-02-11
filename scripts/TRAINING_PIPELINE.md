# 학습 파이프라인 (01~04 + 문서)

## 01. `scripts/01_train_policy.py`
- 목적: 행동 분류 정책 모델 학습
- 입력: `logs/*.jsonl` (기보)
- 출력: `models/policy-model.json` (기본)
- 핵심:
  - `decision_trace`에서 의사결정 샘플 추출
  - 타입별(`play/match/option`) 확률 모델 생성
  - 컨텍스트별/전역 카운트 저장

실행:
```bash
python scripts/01_train_policy.py --input logs/*.jsonl --output models/policy-v1.json
```

## 02. `scripts/02_train_value.py`
- 목적: 골드 기대값 회귀 모델 학습
- 입력: `logs/*.jsonl`
- 출력: `models/value-model.json` (기본)
- 핵심:
  - 상태/행동 특징(해시 벡터) + 숫자 특징으로 선형 회귀
  - 타깃: 턴 행위자 기준 최종 점수차 * `gold_per_point`

실행:
```bash
python scripts/02_train_value.py --input logs/*.jsonl --output models/value-v1.json
```

## 03. `scripts/03_evaluate.py`
- 목적: 구모델 vs 신모델 오프라인 비교 리포트
- 입력:
  - 정책 구/신 모델
  - (선택) 가치 구/신 모델
  - 평가용 `logs/*.jsonl`
- 출력: `logs/model-eval-*.json`
- 핵심:
  - Policy: Top1, NLL, 듀얼 승률
  - Value: MAE/RMSE, 듀얼 승률

실행:
```bash
python scripts/03_evaluate.py \
  --input logs/*.jsonl \
  --policy-old models/policy-v0.json \
  --policy-new models/policy-v1.json \
  --value-old models/value-v0.json \
  --value-new models/value-v1.json \
  --output logs/model-eval-v1.json
```

## 04. `scripts/04_run_pipeline.py`
- 목적: 01 -> 02 -> 03 순차 실행 자동화

실행:
```bash
python scripts/04_run_pipeline.py --input logs/*.jsonl --tag v1
```

## 권장 실행 순서
1. 시뮬레이션 로그 생성
2. `01_train_policy.py`
3. `02_train_value.py`
4. `03_evaluate.py` (또는 `04_run_pipeline.py` 일괄 실행)

## 자가 대결 권장 판수
1. 0단계(로직 검증): `1만 판`
2. 1단계(초기 정책 학습): `50만 판`
3. 2단계(모델1 self-play): `100만 판`
4. 3단계(모델2 고도화): `500만 ~ 1000만 판`
5. 4단계(운영급): `2000만 판+`

평가:
- 학습과 분리된 고정 `10만 판` 평가 세트 유지(시드 고정 권장)

## 참고
- 현재 환경에 Python 실행기가 없으면 스크립트 실행이 실패할 수 있습니다.
- 로그가 클 경우 `--input`을 여러 개로 분할해 학습하는 것을 권장합니다.
