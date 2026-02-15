# 맞고 데모 룰북 (현재 구현 기준)

브라우저에서 실행되는 2인 맞고(플레이어/AI) 데모입니다.
기준 코드: `src/state.js`, `src/engine/*`, `src/cards.js`.

## 한게임 기본룰 별도 문서
- 비교/원문 정리는 `docs/HANGAME_BASIC_RULE.md`

## 1. 빠른 실행
- 개발 서버: `npm run dev`
- 배포 빌드: `npm run build`
- 빌드 미리보기: `npm run preview`

## 2. 게임 개요
- 기본 화투 48장 + 보너스 카드 2장(`Bonus Double`, `Bonus Triple`)
- 시작 배분: 플레이어 10장, AI 10장, 바닥 8장, 나머지 덱
- AI 기본 봇 정책: `heuristic_v3` (모델 정책 선택 가능)

### 2-1. 선턴 결정 (밤일낮장)
- 첫 판: 각자 시작 손패 첫 카드 월 비교
- 밤(18:00~05:59): 낮은 월 선
- 낮(06:00~17:59): 높은 월 선
- 동월이면 랜덤
- 다음 판: 직전 승자 선, 나가리면 직전 승자 유지

### 2-2. 시작 보정
- 손패 보너스는 즉시 획득 후 손패 10장 보충
- 바닥 보너스는 선턴이 즉시 획득
- 가져간 수만큼 덱에서 보충해 바닥 8장 유지

## 3. 대통령/나가리
### 3-1. 손패 대통령
- 조건: 자기 첫 턴 시작 시 같은 월 4장
- 단계: `president-choice`
- 선택: `10점 종료` 또는 `들고치기`

들고치기 승리 보너스:
- 해당 라운드에서 흔들기/폭탄 1회 이상 발생 후 승리 시 최종 배수 x4 추가

### 3-2. 바닥 대통령
- 조건: 시작 바닥패에 같은 월 4장
- 처리: 선공 즉시 승리, 10점 종료

### 3-3. 나가리
- 무승부, 양측 무득점, Go 후 점수 상승 실패 시 나가리
- 다음 판 배수 x2, 연속 나가리면 누적(`x2 -> x4 -> x8 ...`)

## 4. 턴/매치 처리
### 4-1. 기본 턴
1. 손패 1장 낸다
2. 월이 맞으면 캡처, 아니면 바닥에 둔다
3. 덱 1장 뒤집는다
4. 뒤집은 카드도 동일 규칙 처리

### 4-2. 매치 규칙
- 0매치: 바닥 배치
- 1매치: 2장 캡처
- 2매치: 타입(광/열/띠/피) 다르면 선택 UI, 같으면 자동 선택
- 3매치 이상: 싹쓸이 캡처(`ttak +1`)
- 뒤집기 2매치도 동일 규칙

## 5. 이벤트/특수 규칙
- 쪽(`jjob`): 상대 피 1장 강탈
- 뻑(`ppuk`): 뻑 상태 발생 이벤트
- 따닥/판쓸: 상대 피 1장 강탈(턴 종료 시 반영)

### 5-1. 흔들기/폭탄
- 흔들기: 손패 같은 월 3장 + 바닥 0매치 월
- 폭탄: 손패 같은 월 3장 + 바닥 1장
- 흔들기/폭탄 모두 1회당 최종 배수 x2 누적

### 5-2. 뻑 먹기/연뻑
- 자뻑/상대뻑 먹기 동일 처리: 강탈 +1
- 뻑 보류 보너스도 자뻑/일반뻑 동일 회수/강탈
- 첫뻑/연뻑 골드 보상:
- 첫 턴 뻑: 7점
- 1~2턴 연속 뻑: 14점
- 1~3턴 연속 뻑: 21점
- 누적 3뻑: 즉시 승리(7점 처리)

### 5-3. 국진(9월 열) 선택
- 처음 획득 시 기본은 `열`
- 피 계산/점수 판정에서 `열`/`쌍피` 결과가 갈리는 시점에 1회 선택(낙장불입)
- `열`: 열끗 장수/멍박 계산 포함
- `쌍피`: 피 2로 계산
- 쌍피 구성: `11월 쌍피`, `12월 쌍피`, `국진(쌍피 선택 시)`, `Bonus Double(2피)`, `Bonus Triple(3피)`
- 피 강탈 우선순위: `1피 > 2피 > 국진 2피 > 3피`

### 5-4. 기타
- 마지막 손 1장 턴에서는 `뻑/판쓸/쪽/따닥` 미발동
- 손패 소진 진행 필요 시 `Pass` 카드 자동 생성

## 6. 점수 계산
최종점수 = `(기본점수 + Go보너스) x 배수`

### 6-1. 기본점수
- 광: 3광 3점 / 비광포함3광 2점 / 4광 4점 / 5광 15점
- 열끗: 5장 이상 `(장수 - 4)`
- 띠: 5장 이상 `(장수 - 4)`
- 피: 피 합 10 이상 `(피합 - 9)`
- 단/족보:
- 홍단(1/2/3월) +3
- 청단(6/9/10월) +3
- 초단(4/5/7월) +3
- 고도리(2/4/8월 열끗) +5

### 6-2. Go 보너스
- Go 1회당 기본점수 +1 누적

### 6-3. 배수
- 고 배수: `3고 x2, 4고 x4, 5고 x8, 6고 x16 ...` (`goCount>=3`에서 `2^(goCount-2)`)
- 흔들기/폭탄: 각각 1회당 x2 누적
- 독박: STOP 승리 시 패자 `goCount > 0`이면 x2
- 광박: 상대 광 0 + 내 광 3장 이상
- 피박: 상대 피 합 1~7 + 내가 피로 득점 (`0`이면 면박으로 피박 면제)
- 멍박: 상대 열끗 0 + 내 열끗 7장 이상

## 7. Go/Stop 및 라운드 종료
- 7점 이상 + 직전 Go 기준점(`lastGoBase`) 상승 시 Go/Stop
- Go: `goCount +1`, `lastGoBase` 갱신
- Stop: 즉시 종료
- 종료 조건: Stop 또는 양측 손패 0
- 들고치기 x4 조건 충족 시 추가 적용
- 나가리는 승패 없이 종료, 다음 판 누적 배수만 증가

## 8. 골드 정산
- 시작 자금: 각 1,000,000골드
- 점당 단위: `POINT_GOLD_UNIT`(기본 100)
- 승자 정산: `최종점수 x POINT_GOLD_UNIT`
- 패자는 보유 골드 한도 내 지급(음수 불가)
- 지급 후 0골드면 해당 판은 0 유지, 다음 게임 시작 시 1,000,000골드 재시작

## 9. 더미/강탈/피 규칙(최신)
- 더미 패스 카드는 손패에서만 존재 가능(실카드 ID 중복 금지)
- 더미 생성 기준:
- `expectedHand = 10 - turnCount`
- `dummyNeeded = expectedHand - realHandCount` (양수일 때만)
- 더미 사용: 1장만 소모, 뒤집기/매치 판정은 일반 턴과 동일
- 피 강탈 우선순위: `1피 -> 2피 -> 국진쌍피 -> 3피`
- 같은 우선순위면 나중에 획득한 피부터 강탈

## 10. 용어 이중표기(동의어)
- 대통령/`phase: "president-choice"`, `pendingPresident` = 총통
- 폭탄/`events.bomb`, `declareBomb`, `type: "declare_bomb"` = 쿵
- 쪽/`events.jjob` = 키스 = 귀신
- 따닥/`events.ddadak` = 쌍쓸
- 뻑/`events.ppuk`, `ppukState` = 설사 = 뻐꾸기 = 싼다
- 뻑 먹기/`events.jabbeok`
- 판쓸/`events.ssul` = 쓸
- 판쓰리(한게임 표현) = 판쓸
- 국진 쌍피/`gukjinMode: "junk"`, `gukjinLocked`, `gukjinTransformed`
- 수류탄(2장 폭탄): 현재 엔진 미구현

## 11. 모드/기보/시뮬레이션
### 11-1. 플레이 모드
- 사람 vs AI
- 사람 vs 사람
- AI vs AI

### 11-2. 기보 로그
- UI `기보 내보내기`로 JSON 다운로드
- 주요 기록:
- `kibo.turn_end.action.matchEvents` (`source(hand/flip)` + `eventTag`)
- `kibo.turn_end.action.captureBySource` (`hand[]/flip[]`)
- `kibo.turn_end.ppukState`
- 턴 리플레이 지원: 이전/다음/자동재생/슬라이더/속도

## 12. 운영 스크립트
- RAM 가드: `./scripts/run-with-ram-guard.ps1 -- -3 scripts/02_train_value.py ...`
- 챔피언 사이클: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_champion_cycle.ps1 -Champion heuristic_v3 -Challenger heuristic_v3 -GamesPerSide 100000 -PromoteThreshold 0.52 -Rounds 1 -Tag stage4`
- 챔피언 사이클 재개: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_champion_cycle.ps1 -Tag stage4 -Resume`
- fast 루프: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-fast-loop.ps1 -TrainGames 200000 -Workers 4 -Tag fast`
- 리그 fast 루프: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-fast-loop.ps1 -TrainGames 200000 -Workers 4 -Tag fast-league -LeagueConfig scripts/league_config.example.json`
- 점수표 갱신: `py -3 scripts/update_champion_scoreboard.py`

## 13. React 구조
- UI 프레임워크: React + Vite
- 엔진: `src/gameEngine.js`, `src/state.js`, `src/engine/*`
- UI 컴포넌트:
- `src/ui/components/GameBoard.jsx`
- `src/ui/components/GameOverlays.jsx`
- `src/ui/components/CardView.jsx`

## 14. 레이아웃 디버그(임시)
- `styles.css`의 `TEMP: layout debug rainbow borders` 블록이 켜져 있으면 무지개 테두리 표시
- 릴리즈 전 해당 블록 삭제
