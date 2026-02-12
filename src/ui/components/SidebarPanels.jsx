import { useMemo, useRef, useState } from "react";
import { buildDeck } from "../../cards.js";
import { POINT_GOLD_UNIT } from "../../engine/economy.js";
import CardView from "./CardView.jsx";

export default function SidebarPanels({
  state,
  ui,
  setUi,
  startGame,
  onStartSpecifiedGame,
  onStartRandomGame,
  participantType,
  safeLoadJson,
  replayModeEnabled,
  onLoadReplay,
  onClearReplay
}) {
  const fileInputRef = useRef(null);
  const [showCardGuide, setShowCardGuide] = useState(false);

  const monthCards = useMemo(() => {
    const order = { kwang: 0, five: 1, ribbon: 2, junk: 3 };
    return buildDeck()
      .filter((c) => c.month >= 1 && c.month <= 12)
      .sort((a, b) => a.month - b.month || order[a.category] - order[b.category] || a.id.localeCompare(b.id));
  }, []);

  const bonusCards = useMemo(() => buildDeck().filter((c) => c.month === 13), []);

  const dummyCard = {
    id: "guide-pass",
    month: 0,
    name: "Dummy Pass",
    asset: "/cards/pass.svg",
    passCard: true
  };
  const backCard = { id: "guide-back", month: 0, name: "Back", asset: "/cards/back.svg" };
  const deckCard = { id: "guide-deck", month: 0, name: "Deck", asset: "/cards/deck-stack.svg" };

  const hiddenLogPatterns = ["게임 시작 - 룰:", "선턴 유지:", "매치 캡처", "카드 선택"];
  const visibleLogs = state.log.filter(
    (line) => !hiddenLogPatterns.some((pattern) => String(line).includes(pattern))
  );

  const extractReplayEntry = (parsed) => {
    if (Array.isArray(parsed)) {
      for (let i = parsed.length - 1; i >= 0; i -= 1) {
        const item = parsed[i];
        if (item && Array.isArray(item.kibo)) return item;
      }
      return null;
    }
    if (parsed && Array.isArray(parsed.kibo)) return parsed;
    return null;
  };

  const loadReplayFromFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result || ""));
        const entry = extractReplayEntry(parsed);
        if (!entry) {
          window.alert("기보 형식을 찾지 못했습니다. kibo 배열이 있는 JSON 파일을 선택하세요.");
          return;
        }
        onLoadReplay(entry, file.name);
      } catch {
        window.alert("JSON 파싱에 실패했습니다. 파일 내용을 확인하세요.");
      } finally {
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    };
    reader.readAsText(file, "utf-8");
  };

  return (
    <div>
      <div className="panel controls">
        <div className="controls-section">
          <div className="control-row">
            <button onClick={() => window.open("./rules/index.html", "_blank")}>룰페이지 열기</button>
            <button onClick={() => setShowCardGuide((v) => !v)}>{showCardGuide ? "패보기 닫기" : "패보기"}</button>
          </div>
        </div>
        <div className="controls-section">
          <div className="meta-row">
            <span className="meta-label">현재 턴</span>
            <span>{state.players[state.currentTurn].label}</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">덱 잔여</span>
            <span>{state.deck.length}장</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">시드</span>
            <span>{ui.seed}</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">점당 골드</span>
            <span>{POINT_GOLD_UNIT}골드</span>
          </div>
        </div>

        <div className="controls-section">
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

        <div className="controls-section">
          <div className="control-row">
            <input
              className="seed-input"
              value={ui.seed}
              onChange={(e) => setUi((u) => ({ ...u, seed: e.target.value }))}
            />
            <button onClick={onStartSpecifiedGame}>지정 게임시작</button>
          </div>
          <div className="control-row">
            <button onClick={onStartRandomGame}>새 게임시작</button>
          </div>
        </div>

        <div className="controls-section">
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
      </div>

      {showCardGuide && (
        <div className="panel card-guide-panel">
          <div className="section-title">패보기</div>
          <div className="meta">초보자용 카드 목록: 1~12월 기본패, 보너스패, 더미패, 뒤집은 패, 덱</div>

          <div className="card-guide-section">
            <div className="meta">1~12월 기본패 (월/종류 순)</div>
            <div className="card-guide-grid card-guide-grid-main">
              {monthCards.map((card) => (
                <div key={`guide-main-${card.id}`} className="card-guide-item">
                  <CardView card={card} />
                  <div className="card-guide-label">{card.month}월 {card.category}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card-guide-section">
            <div className="meta">특수/상태 카드</div>
            <div className="card-guide-grid card-guide-grid-small">
              {bonusCards.map((card) => (
                <div key={`guide-bonus-${card.id}`} className="card-guide-item">
                  <CardView card={card} />
                  <div className="card-guide-label">보너스패</div>
                </div>
              ))}
              <div className="card-guide-item">
                <CardView card={dummyCard} />
                <div className="card-guide-label">더미패</div>
              </div>
              <div className="card-guide-item">
                <CardView card={backCard} />
                <div className="card-guide-label">뒤집은 패</div>
              </div>
              <div className="card-guide-item">
                <CardView card={deckCard} />
                <div className="card-guide-label">덱</div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="panel">
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          style={{ display: "none" }}
          onChange={loadReplayFromFile}
        />
        <button onClick={() => fileInputRef.current?.click()}>기보 불러오기</button>
        {replayModeEnabled && <button onClick={onClearReplay}>불러온 기보 닫기</button>}
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

      <div className="panel log">
        {visibleLogs.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
    </div>
  );
}


