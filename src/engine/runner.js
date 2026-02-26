/* ============================================================================
 * Auto-run helpers
 * - Resolve actor by phase
 * - Advance bot-vs-bot turns until stop condition
 * ========================================================================== */

const DEFAULT_MAX_AUTO_STEPS = 400;

export function getActionPlayerKey(state) {
  if (state.phase === "playing") return state.currentTurn;
  if (state.phase === "go-stop") return state.pendingGoStop || null;
  if (state.phase === "select-match") return state.pendingMatch?.playerKey || null;
  if (state.phase === "president-choice") return state.pendingPresident?.playerKey || null;
  if (state.phase === "shaking-confirm") return state.pendingShakingConfirm?.playerKey || null;
  if (state.phase === "gukjin-choice") return state.pendingGukjinChoice?.playerKey || null;
  return null;
}

export function advanceAutoTurns(state, isBotPlayer, playBot, maxSteps = DEFAULT_MAX_AUTO_STEPS) {
  let next = state;
  for (let i = 0; i < maxSteps; i += 1) {
    const actor = getActionPlayerKey(next);
    if (!actor || !isBotPlayer(actor)) break;
    const updated = playBot(next, actor);
    if (updated === next) break;
    next = updated;
    if (next.phase === "resolution") break;
  }
  return next;
}
