import CardView from "./CardView.jsx";

export default function SidebarPanels({
  state,
  ui,
  setUi,
  setState,
  startGame,
  runAuto,
  randomSeed,
  participantType,
  safeLoadJson,
  replayFrames,
  replayIdx,
  replayFrame,
  formatActionText,
  formatEventsText,
  remain,
  actions
}) {
  const savedLogs = safeLoadJson("kflower_game_logs", []);

  return (
    <div>
      <div className="panel controls">
        <div>현재 턴: {state.players[state.currentTurn].label}</div>
        <div className="meta">덱 잔여: {state.deck.length}장</div>
        <div className="meta">시드: {ui.seed} / 누적배수: x{ui.carryOverMultiplier}</div>
        <div className="control-row">
          <input value={ui.seed} onChange={(e) => setUi((u) => ({ ...u, seed: e.target.value }))} />
          <button onClick={() => startGame()}>시드 새 게임</button>
        </div>
        <button
          onClick={() => {
            setUi((u) => ({ ...u, seed: randomSeed(), lastWinnerKey: null }));
            setTimeout(() => startGame(), 0);
          }}
        >
          랜덤 시드
        </button>
        <button onClick={() => setState((prev) => runAuto(prev, ui, true))}>AI 턴 진행</button>
        <button
          onClick={() => {
            setUi((u) => ({ ...u, lastWinnerKey: null }));
            setTimeout(() => startGame(), 0);
          }}
        >
          새 게임
        </button>
        <div className="toggle-list">
          <label className="toggle-item">
            <span>진행 속도</span>
            <select
              value={ui.speedMode || "fast"}
              onChange={(e) => setUi((u) => ({ ...u, speedMode: e.target.value }))}
            >
              <option value="fast">Fast (즉시)</option>
              <option value="visual">Visual (턴 애니메이션)</option>
            </select>
          </label>
          <label className="toggle-item">
            <span>Visual 지연</span>
            <select
              value={ui.visualDelayMs || 400}
              onChange={(e) => setUi((u) => ({ ...u, visualDelayMs: Number(e.target.value) }))}
              disabled={(ui.speedMode || "fast") !== "visual"}
            >
              <option value={150}>0.15s</option>
              <option value={300}>0.30s</option>
              <option value={400}>0.40s</option>
              <option value={700}>0.70s</option>
              <option value={1000}>1.00s</option>
            </select>
          </label>
          <label className="toggle-item">
            <input
              type="checkbox"
              checked={ui.revealAiHand}
              onChange={(e) => setUi((u) => ({ ...u, revealAiHand: e.target.checked }))}
            />
            <span>AI 패 공개</span>
          </label>
          <label className="toggle-item">
            <input
              type="checkbox"
              checked={ui.sortHand}
              onChange={(e) => setUi((u) => ({ ...u, sortHand: e.target.checked }))}
            />
            <span>패 정렬</span>
          </label>
        </div>
      </div>

      <div className="panel">
        <div className="section-title">플레이 모드</div>
        <select
          value={
            participantType(ui, "human") === "human" && participantType(ui, "ai") === "ai"
              ? "hva"
              : participantType(ui, "human") === "human" && participantType(ui, "ai") === "human"
              ? "hvh"
              : "ava"
          }
          onChange={(e) => {
            const value = e.target.value;
            setUi((u) => ({
              ...u,
              participants:
                value === "hvh"
                  ? { human: "human", ai: "human" }
                  : value === "ava"
                  ? { human: "ai", ai: "ai" }
                  : { human: "human", ai: "ai" },
              lastWinnerKey: null
            }));
            setTimeout(() => startGame(), 0);
          }}
        >
          <option value="hva">사람 vs AI</option>
          <option value="hvh">사람 vs 사람</option>
          <option value="ava">AI vs AI</option>
        </select>
      </div>

      <div className="panel">
        <div className="section-title">기보 로그</div>
        <div className="meta">저장됨: {savedLogs.length}판 (브라우저 로컬)</div>
        <button
          onClick={() => {
            const logs = safeLoadJson("kflower_game_logs", []);
            const blob = new Blob([JSON.stringify(logs, null, 2)], { type: "application/json" });
            const a = document.createElement("a");
            const ts = new Date().toISOString().replace(/[:.]/g, "-");
            a.href = URL.createObjectURL(blob);
            a.download = `kibo-${ts}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(a.href);
          }}
        >
          기보 내보내기
        </button>
        <button
          onClick={() => {
            localStorage.removeItem("kflower_game_logs");
            setUi((u) => ({ ...u }));
          }}
        >
          기보 초기화
        </button>
      </div>

      <div className="panel">
        <div className="section-title">턴 리플레이</div>
        {replayFrame ? (
          <>
            <div className="meta">상태: {ui.replay.enabled ? "켜짐" : "꺼짐"} / 프레임 {replayIdx + 1} / {replayFrames.length}</div>
            <div className="meta">턴: {replayFrame.turnNo} / 행동자: {replayFrame.actor ? state.players[replayFrame.actor].label : "-"} / 덱: {replayFrame.deckCount}장</div>
            <div className="meta">행동: {formatActionText(replayFrame.action)}</div>
            <div className="meta">뒤집기: {(replayFrame.action?.flips || []).map((c) => `${c.month}월 ${c.name}`).join(", ") || "-"}</div>
            <div className="meta">강탈: 피 {replayFrame.steals?.pi || 0}, 골드 {replayFrame.steals?.gold || 0}</div>
            <div className="meta">이벤트: {formatEventsText(replayFrame.events)}</div>
            <div className="control-row">
              <button onClick={() => setUi((u) => ({ ...u, replay: { ...u.replay, enabled: !u.replay.enabled, autoPlay: false } }))}>
                {ui.replay.enabled ? "리플레이 종료" : "리플레이 시작"}
              </button>
              <button
                disabled={!ui.replay.enabled || replayIdx <= 0}
                onClick={() => setUi((u) => ({ ...u, replay: { ...u.replay, turnIndex: Math.max(0, u.replay.turnIndex - 1) } }))}
              >
                이전 턴
              </button>
              <button
                disabled={!ui.replay.enabled || replayIdx >= replayFrames.length - 1}
                onClick={() => setUi((u) => ({ ...u, replay: { ...u.replay, turnIndex: Math.min(replayFrames.length - 1, u.replay.turnIndex + 1) } }))}
              >
                다음 턴
              </button>
              <button
                disabled={!ui.replay.enabled || replayFrames.length <= 1}
                onClick={() => setUi((u) => ({ ...u, replay: { ...u.replay, autoPlay: !u.replay.autoPlay } }))}
              >
                {ui.replay.autoPlay ? "자동재생 정지" : "자동재생"}
              </button>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(0, replayFrames.length - 1)}
              value={replayIdx}
              disabled={!ui.replay.enabled}
              onChange={(e) => setUi((u) => ({ ...u, replay: { ...u.replay, turnIndex: Number(e.target.value) } }))}
            />
            <div className="control-row">
              <div className="meta">자동재생 간격</div>
              <select
                value={ui.replay.intervalMs}
                disabled={!ui.replay.enabled}
                onChange={(e) => setUi((u) => ({ ...u, replay: { ...u.replay, intervalMs: Number(e.target.value) } }))}
              >
                <option value={500}>0.5s</option>
                <option value={900}>0.9s</option>
                <option value={1300}>1.3s</option>
                <option value={1800}>1.8s</option>
              </select>
            </div>
            <div className="replay-stage">
              <div className="meta">보드</div>
              <div className="center-row">{(replayFrame.board || []).map((c) => <CardView key={`rb-${c.id}`} card={c} />)}</div>
              <div className="meta">{state.players.human.label} 손패</div>
              <div className="center-row">{(replayFrame.hands?.human || []).map((c) => <CardView key={`rh-${c.id}`} card={c} />)}</div>
              <div className="meta">{state.players.ai.label} 손패</div>
              <div className="center-row">{(replayFrame.hands?.ai || []).map((c) => <CardView key={`ra-${c.id}`} card={c} />)}</div>
            </div>
          </>
        ) : (
          <div className="meta">기보가 없어 리플레이를 시작할 수 없습니다.</div>
        )}
      </div>

      {state.phase === "playing" && participantType(ui, state.currentTurn) === "human" && (
        <div className="panel">
          <div className="section-title">선언</div>
          <div className="meta">선언 후 해당 카드를 바로 내지 않아도 됩니다.</div>
          <div className="control-row">
            {actions.getDeclarableShakingMonths(state, state.currentTurn).map((m) => (
              <button key={`s-${m}`} onClick={() => actions.declareShaking(m)}>흔들기 {m}월</button>
            ))}
            {actions.getDeclarableBombMonths(state, state.currentTurn).map((m) => (
              <button key={`b-${m}`} onClick={() => actions.declareBomb(m)}>폭탄 {m}월</button>
            ))}
          </div>
        </div>
      )}

      {state.phase === "go-stop" && state.pendingGoStop && participantType(ui, state.pendingGoStop) === "human" && (
        <div className="panel">
          <div className="section-title">Go / Stop</div>
          <button onClick={() => actions.chooseGo(state.pendingGoStop)}>Go</button>
          <button onClick={() => actions.chooseStop(state.pendingGoStop)}>Stop</button>
        </div>
      )}

      {state.phase === "president-choice" && state.pendingPresident?.playerKey && participantType(ui, state.pendingPresident.playerKey) === "human" && (
        <div className="panel">
          <div className="section-title">대통령 선택</div>
          <button onClick={() => actions.choosePresidentStop(state.pendingPresident.playerKey)}>10점 종료</button>
          <button onClick={() => actions.choosePresidentHold(state.pendingPresident.playerKey)}>들고치기</button>
        </div>
      )}

      {state.phase === "kung-choice" && state.pendingKung?.playerKey && participantType(ui, state.pendingKung.playerKey) === "human" && (
        <div className="panel">
          <div className="section-title">쿵 선택</div>
          <button onClick={() => actions.chooseKungUse(state.pendingKung.playerKey)}>쿵 사용</button>
          <button onClick={() => actions.chooseKungPass(state.pendingKung.playerKey)}>패스</button>
        </div>
      )}

      {state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey && participantType(ui, state.pendingGukjinChoice.playerKey) === "human" && (
        <div className="panel">
          <div className="section-title">국진 선택</div>
          <button onClick={() => actions.chooseGukjinMode(state.pendingGukjinChoice.playerKey, "five")}>열로 확정</button>
          <button onClick={() => actions.chooseGukjinMode(state.pendingGukjinChoice.playerKey, "junk")}>쌍피로 확정</button>
        </div>
      )}

      <div className="panel">
        <div className="section-title">국진(9월 열) 처리</div>
        <div className="meta">
          플레이어: {state.players.human.gukjinMode === "junk" ? "쌍피" : "열"} ({state.players.human.gukjinLocked ? "확정" : "미확정"}) / AI: {state.players.ai.gukjinMode === "junk" ? "쌍피" : "열"} ({state.players.ai.gukjinLocked ? "확정" : "미확정"})
        </div>
      </div>

      <div className="panel">
        <div className="section-title">남은 월 카드(전체 기준)</div>
        <div className="meta">{Object.entries(remain).map(([m, c]) => `${m}월: ${c}장`).join(" · ")}</div>
      </div>

      <div className="panel log">
        {state.log.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>

      <div className="panel">
        <div className="section-title">룰/변형 참고</div>
        <button onClick={() => window.open("./rules/index.html", "_blank")}>룰 페이지 열기</button>
      </div>
    </div>
  );
}
