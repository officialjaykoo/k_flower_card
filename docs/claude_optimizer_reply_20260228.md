# Claude 의견 답변 (GPT)

결론: Claude가 제시한 원인 분석은 1번은 타당하지만, 첨부된 최종 코드에는 2번이 코드상 미완료입니다.

## 판단
- 동의: `block/expand` 방향 기준을 `baseline_ev - zone_ev`로 바꾸는 것은 STOP 희소 문제를 줄이는 올바른 방향입니다.
- 반박: 첨부본 `greedyPlan`의 expand 경로는 여전히 `subStop.length`를 요구하므로 STOP 데이터 의존이 남아 있습니다.
- 보완: expand를 포함한 greedy 전체를 GO 레코드 기반으로 재점수화하고, 스텝마다 재탐색 후 선택된 존을 제거해야 중복 왜곡을 줄일 수 있습니다.

## 이번에 만든 대응 파일
- `scripts/optimizer_by_cl.mjs`

핵심 차이:
- discover + greedy 모두 STOP-free EV gap(`baseline_ev - zone_ev`) 일관 적용
- expand에서 STOP 샘플 최소치 조건 제거
- greedy에서 매 스텝 zone 재발견 + 선택 zone 제거로 overlap 왜곡 완화
- 파라미터 매핑은 기존 휴리스틱 키 참조 여부를 검사해 fail-fast 유지

## 유의
- expand는 실제 STOP 분포를 직접 관측하지 않으므로 과대추정 위험이 있습니다.
- 이 파일은 expand 추정치에 보수 가중(`expand_flip_weight`)을 적용했습니다.
- 최종 채택은 반드시 1000게임 실측 결과로 확인해야 합니다.
