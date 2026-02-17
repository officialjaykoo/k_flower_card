import { useRef } from "react";

export default function GameBoardControlsPanel({
  state,
  ui,
  setUi,
  participantType,
  onStartSpecifiedGame,
  onStartRandomGame,
  onLoadReplay,
  onClearReplay,
  onOpenRulesPopup,
  onOpenCardGuidePopup,
  onOpenGameLogPopup,
  t,
  supportedLanguages = ["ko", "en"]
}) {
  const replayFileInputRef = useRef(null);

  const loadReplayFromFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result || ""));
        if (!parsed || !Array.isArray(parsed.kibo)) {
          window.alert(t("controls.alert.invalidReplay"));
          return;
        }
        onLoadReplay(parsed, file.name);
      } catch {
        window.alert(t("controls.alert.invalidJson"));
      } finally {
        if (replayFileInputRef.current) replayFileInputRef.current.value = "";
      }
    };
    reader.readAsText(file, "utf-8");
  };

  const exportKiboJson = () => {
    const snapshot = {
      recordedAt: new Date().toISOString(),
      seed: ui.seed,
      ruleKey: state.ruleKey,
      participants: { ...ui.participants },
      winner: state.result?.winner || null,
      nagari: state.result?.nagari || false,
      nagariReasons: state.result?.nagariReasons || [],
      totals: {
        human: state.result?.human?.total ?? null,
        ai: state.result?.ai?.total ?? null
      },
      startingTurnKey: state.startingTurnKey,
      endingTurnKey: state.currentTurn,
      log: state.log.slice(),
      players: state.players,
      kibo: state.kibo || []
    };
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = URL.createObjectURL(blob);
    a.download = `kibo-${ts}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  };

  const resetKiboLogs = () => {
    onClearReplay();
    window.alert(t("controls.alert.resetReplayOnly"));
  };

  return (
    <div className="panel controls right-controls-panel">
      <input
        ref={replayFileInputRef}
        type="file"
        accept=".json,application/json"
        style={{ display: "none" }}
        onChange={loadReplayFromFile}
      />
      <div className="controls-section">
        <div className="control-row">
          <button onClick={onOpenRulesPopup}>{t("controls.button.rules")}</button>
          <button onClick={onOpenCardGuidePopup}>{t("controls.button.cardGuide")}</button>
          <button onClick={onOpenGameLogPopup}>{t("controls.button.gameLog")}</button>
          <select
            aria-label={t("controls.language")}
            value={ui.language || "ko"}
            onChange={(e) => setUi((u) => ({ ...u, language: e.target.value }))}
          >
            {supportedLanguages.map((lang) => (
              <option key={`lang-top-${lang}`} value={lang}>
                {t(`controls.language.${lang}`, {}, lang)}
              </option>
            ))}
          </select>
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
                  : { human: "human", ai: "ai" }
            }));
          }}
        >
          <option value="hva">{t("controls.mode.hva")}</option>
          <option value="hvh">{t("controls.mode.hvh")}</option>
          <option value="ava">{t("controls.mode.ava")}</option>
        </select>
      </div>
      <div className="controls-section">
        <div className="control-row seed-row">
          <input
            className="seed-input"
            value={ui.seed}
            onChange={(e) => setUi((u) => ({ ...u, seed: e.target.value }))}
          />
          <button className="btn-start-specified" onClick={onStartSpecifiedGame}>
            {t("controls.button.startSpecified")}
          </button>
        </div>
        <div className="control-row">
          <button onClick={onStartRandomGame}>{t("controls.button.startRandom")}</button>
          <select
            aria-label={t("controls.cardTheme")}
            value={ui.cardTheme || "original"}
            onChange={(e) => setUi((u) => ({ ...u, cardTheme: e.target.value }))}
          >
            <option value="original">{t("controls.cardTheme.original")}</option>
            <option value="k-flower">{t("controls.cardTheme.kFlower")}</option>
          </select>
        </div>
      </div>
      <div className="controls-section">
        <div className="control-row">
          <button onClick={() => replayFileInputRef.current?.click()}>{t("controls.button.loadReplay")}</button>
          <button onClick={exportKiboJson}>{t("controls.button.exportReplay")}</button>
          <button onClick={resetKiboLogs}>{t("controls.button.resetReplay")}</button>
        </div>
      </div>
      <div className="controls-section">
        <div className="toggle-list">
          <label className="toggle-item">
            <span>{t("controls.speed")}</span>
            <select
              value={ui.speedMode || "fast"}
              onChange={(e) => setUi((u) => ({ ...u, speedMode: e.target.value }))}
            >
              <option value="fast">{t("controls.speed.fast")}</option>
              <option value="visual">{t("controls.speed.visual")}</option>
            </select>
          </label>
          <label className="toggle-item">
            <span>{t("controls.visualDelay")}</span>
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
            <span>{t("controls.revealAiHand")}</span>
          </label>
          <label className="toggle-item">
            <input
              type="checkbox"
              checked={ui.fixedHand}
              onChange={(e) => setUi((u) => ({ ...u, fixedHand: e.target.checked }))}
            />
            <span>{t("controls.fixedHand")}</span>
          </label>
        </div>
      </div>
    </div>
  );
}
