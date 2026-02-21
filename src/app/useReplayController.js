import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildReplayFrames } from "../ui/utils/replay.js";

export function useReplayController({ ui, setUi, state, t }) {
  const [loadedReplay, setLoadedReplay] = useState(null);
  const timerRef = useRef(null);

  const replaySource = loadedReplay
    ? {
        kibo: loadedReplay.kibo || [],
        players: loadedReplay.players || state.players
      }
    : state;
  const replayFrames = useMemo(() => buildReplayFrames(replaySource), [replaySource]);
  const replayIdx = replayFrames.length
    ? Math.max(0, Math.min(replayFrames.length - 1, ui.replay.turnIndex))
    : 0;
  const replayFrame = replayFrames[replayIdx] || null;

  useEffect(() => {
    if (!ui.replay.autoPlay || !ui.replay.enabled) return;
    const frames = replayFrames;
    if (frames.length <= 1) return;
    timerRef.current = setInterval(() => {
      setUi((nextUi) => {
        const next = Math.min(frames.length - 1, nextUi.replay.turnIndex + 1);
        const ended = next >= frames.length - 1;
        return {
          ...nextUi,
          replay: {
            ...nextUi.replay,
            turnIndex: next,
            autoPlay: ended ? false : nextUi.replay.autoPlay
          }
        };
      });
    }, ui.replay.intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [ui.replay.autoPlay, ui.replay.enabled, ui.replay.intervalMs, replayFrames, setUi]);

  const onReplayToggle = useCallback(() => {
    setUi((nextUi) => ({
      ...nextUi,
      replay: { ...nextUi.replay, enabled: !nextUi.replay.enabled, autoPlay: false }
    }));
  }, [setUi]);

  const onReplayPrev = useCallback(() => {
    setUi((nextUi) => ({
      ...nextUi,
      replay: { ...nextUi.replay, turnIndex: Math.max(0, nextUi.replay.turnIndex - 1) }
    }));
  }, [setUi]);

  const onReplayNext = useCallback(() => {
    setUi((nextUi) => ({
      ...nextUi,
      replay: {
        ...nextUi.replay,
        turnIndex: Math.min(replayFrames.length - 1, nextUi.replay.turnIndex + 1)
      }
    }));
  }, [replayFrames.length, setUi]);

  const onReplayAutoToggle = useCallback(() => {
    setUi((nextUi) => ({
      ...nextUi,
      replay: { ...nextUi.replay, autoPlay: !nextUi.replay.autoPlay }
    }));
  }, [setUi]);

  const onReplaySeek = useCallback(
    (idx) => {
      setUi((nextUi) => ({ ...nextUi, replay: { ...nextUi.replay, turnIndex: Number(idx) } }));
    },
    [setUi]
  );

  const onReplayIntervalChange = useCallback(
    (ms) => {
      setUi((nextUi) => ({ ...nextUi, replay: { ...nextUi.replay, intervalMs: Number(ms) } }));
    },
    [setUi]
  );

  const onLoadReplay = useCallback(
    (entry, label = t("replay.loadedSourceDefault")) => {
      setLoadedReplay({
        label,
        kibo: Array.isArray(entry?.kibo) ? entry.kibo : [],
        players: entry?.players || state.players
      });
      setUi((nextUi) => ({
        ...nextUi,
        replay: { ...nextUi.replay, enabled: true, autoPlay: false, turnIndex: 0 }
      }));
    },
    [setUi, state.players, t]
  );

  const onClearReplay = useCallback(() => {
    setLoadedReplay(null);
    setUi((nextUi) => ({
      ...nextUi,
      replay: { ...nextUi.replay, enabled: false, autoPlay: false, turnIndex: 0 }
    }));
  }, [setUi]);

  return {
    replayModeEnabled: !!loadedReplay,
    replaySourceLabel: loadedReplay?.label || null,
    replayPlayers: loadedReplay?.players || state.players,
    replayFrames,
    replayIdx,
    replayFrame,
    onReplayToggle,
    onReplayPrev,
    onReplayNext,
    onReplayAutoToggle,
    onReplaySeek,
    onReplayIntervalChange,
    onLoadReplay,
    onClearReplay
  };
}

