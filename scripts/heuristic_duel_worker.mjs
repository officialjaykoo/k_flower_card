import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
  calculateScore,
  scoringPiCount,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  playTurn,
  chooseMatch,
  chooseGo,
  chooseStop,
  chooseShakingYes,
  chooseShakingNo,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
} from "../src/engine/index.js";
import { createWriteStream, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { getActionPlayerKey } from "../src/engine/runner.js";
import { aiPlay } from "../src/ai/aiPlay.js";
import { BOT_POLICIES, normalizeBotPolicy } from "../src/ai/policies.js";

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    policyA: "",
    policyB: "",
    games: 1000,
    seed: "heuristic-duel",
    maxSteps: 600,
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
    kiboDetail: "lean",
    kiboOut: "",
    datasetOut: "",
    datasetActor: "all",
    unresolvedOut: "",
    unresolvedLimit: 0,
  };

  while (args.length > 0) {
    const raw = String(args.shift() || "");
    if (!raw.startsWith("--")) throw new Error(`Unknown argument: ${raw}`);
    const eq = raw.indexOf("=");
    let key = raw;
    let value = "";
    if (eq >= 0) {
      key = raw.slice(0, eq);
      value = raw.slice(eq + 1);
    } else {
      value = String(args.shift() || "");
    }

    if (key === "--policy-a") out.policyA = String(value || "").trim().toLowerCase();
    else if (key === "--policy-b") out.policyB = String(value || "").trim().toLowerCase();
    else if (key === "--games") out.games = Math.max(1, Number(value || 1000));
    else if (key === "--seed") out.seed = String(value || "heuristic-duel");
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Number(value || 600));
    else if (key === "--first-turn-policy") out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    else if (key === "--fixed-first-turn") out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    else if (key === "--continuous-series") out.continuousSeries = !(String(value || "1").trim() === "0");
    else if (key === "--kibo-detail") out.kiboDetail = String(value || "lean").trim().toLowerCase();
    else if (key === "--kibo-out") out.kiboOut = String(value || "").trim();
    else if (key === "--dataset-out") out.datasetOut = String(value || "").trim();
    else if (key === "--dataset-actor") out.datasetActor = String(value || "all").trim().toLowerCase();
    else if (key === "--unresolved-out") out.unresolvedOut = String(value || "").trim();
    else if (key === "--unresolved-limit")
      out.unresolvedLimit = Math.max(0, Math.floor(Number(value || 0)));
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.policyA) throw new Error("--policy-a is required");
  if (!out.policyB) throw new Error("--policy-b is required");
  if (!BOT_POLICIES.includes(out.policyA)) {
    throw new Error(`invalid --policy-a: ${out.policyA} (allowed: ${BOT_POLICIES.join(", ")})`);
  }
  if (!BOT_POLICIES.includes(out.policyB)) {
    throw new Error(`invalid --policy-b: ${out.policyB} (allowed: ${BOT_POLICIES.join(", ")})`);
  }
  out.policyA = normalizeBotPolicy(out.policyA);
  out.policyB = normalizeBotPolicy(out.policyB);

  if (Math.floor(out.games) !== 1000) {
    throw new Error("this worker is fixed to --games 1000");
  }
  out.games = 1000;

  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human" && out.fixedFirstTurn !== "ai") {
    throw new Error(`invalid --fixed-first-turn: ${out.fixedFirstTurn}`);
  }
  if (out.kiboDetail !== "lean" && out.kiboDetail !== "full") {
    throw new Error(`invalid --kibo-detail: ${out.kiboDetail} (allowed: lean, full)`);
  }
  if (out.datasetActor !== "all" && out.datasetActor !== "human" && out.datasetActor !== "ai") {
    throw new Error(`invalid --dataset-actor: ${out.datasetActor} (allowed: all, human, ai)`);
  }
  if (!Number.isFinite(out.unresolvedLimit) || out.unresolvedLimit < 0) {
    throw new Error(`invalid --unresolved-limit: ${out.unresolvedLimit} (expected >= 0)`);
  }

  return out;
}

function canonicalOptionAction(action) {
  const a = String(action || "").trim();
  if (!a) return "";
  const aliases = {
    choose_go: "go",
    choose_stop: "stop",
    choose_shaking_yes: "shaking_yes",
    choose_shaking_no: "shaking_no",
    choose_president_stop: "president_stop",
    choose_president_hold: "president_hold",
    choose_five: "five",
    choose_junk: "junk",
  };
  return aliases[a] || a;
}

function normalizeOptionCandidates(items) {
  if (!Array.isArray(items)) return [];
  const out = [];
  const seen = new Set();
  for (const raw of items) {
    const v = canonicalOptionAction(raw);
    if (!v || seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

function selectPool(state, actor) {
  if (state.phase === "playing" && state.currentTurn === actor) {
    return { cards: (state.players?.[actor]?.hand || []).map((c) => c.id) };
  }
  if (state.phase === "select-match" && state.pendingMatch?.playerKey === actor) {
    return { boardCardIds: state.pendingMatch.boardCardIds || [] };
  }
  if (state.phase === "go-stop" && state.pendingGoStop === actor) {
    return { options: ["go", "stop"] };
  }
  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === actor) {
    return { options: ["president_stop", "president_hold"] };
  }
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === actor) {
    return { options: ["five", "junk"] };
  }
  if (state.phase === "shaking-confirm" && state.pendingShakingConfirm?.playerKey === actor) {
    return { options: ["shaking_yes", "shaking_no"] };
  }
  return {};
}

function legalCandidatesForDecision(sp, decisionType) {
  if (decisionType === "play") {
    return (sp.cards || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "match") {
    return (sp.boardCardIds || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "option") {
    return normalizeOptionCandidates(sp.options || []);
  }
  return [];
}

function applyAction(state, actor, decisionType, rawAction) {
  let action = String(rawAction || "").trim();
  if (!action) return state;
  if (decisionType === "play") return playTurn(state, action);
  if (decisionType === "match") return chooseMatch(state, action);
  if (decisionType !== "option") return state;

  action = canonicalOptionAction(action);
  if (action === "go") return chooseGo(state, actor);
  if (action === "stop") return chooseStop(state, actor);
  if (action === "shaking_yes") return chooseShakingYes(state, actor);
  if (action === "shaking_no") return chooseShakingNo(state, actor);
  if (action === "president_stop") return choosePresidentStop(state, actor);
  if (action === "president_hold") return choosePresidentHold(state, actor);
  if (action === "five" || action === "junk") return chooseGukjinMode(state, actor, action);
  return state;
}

function stateProgressKey(state) {
  if (!state) return "null";
  const hh = Number(state?.players?.human?.hand?.length || 0);
  const ah = Number(state?.players?.ai?.hand?.length || 0);
  const d = Number(state?.deck?.length || 0);
  return [
    String(state.phase || ""),
    String(state.currentTurn || ""),
    String(state.pendingGoStop || ""),
    String(state.pendingMatch?.stage || ""),
    String(state.pendingPresident?.playerKey || ""),
    String(state.pendingShakingConfirm?.playerKey || ""),
    String(state.pendingGukjinChoice?.playerKey || ""),
    String(state.turnSeq || 0),
    String(state.kiboSeq || 0),
    String(hh),
    String(ah),
    String(d),
  ].join("|");
}

function clamp01(x) {
  const v = Number(x || 0);
  if (v <= 0) return 0;
  if (v >= 1) return 1;
  return v;
}

function tanhNorm(x, scale) {
  const s = Math.max(1e-6, Number(scale || 1));
  return Math.tanh(Number(x || 0) / s);
}

function findCardById(cards, cardId) {
  const id = String(cardId || "");
  if (!Array.isArray(cards)) return null;
  return cards.find((c) => String(c?.id || "") === id) || null;
}

function optionCode(action) {
  const a = canonicalOptionAction(action);
  const map = {
    go: 1,
    stop: 2,
    shaking_yes: 3,
    shaking_no: 4,
    president_stop: 5,
    president_hold: 6,
    five: 7,
    junk: 8,
  };
  return Number(map[a] || 0) / 8.0;
}

function candidateCard(state, actor, decisionType, candidate) {
  if (decisionType === "play") {
    return findCardById(state?.players?.[actor]?.hand || [], candidate);
  }
  if (decisionType === "match") {
    return findCardById(state?.board || [], candidate);
  }
  return null;
}

function countCardsByMonth(cards, month) {
  const targetMonth = Number(month || 0);
  if (!Array.isArray(cards) || targetMonth <= 0) return 0;
  let count = 0;
  for (const card of cards) {
    if (Number(card?.month || 0) === targetMonth) count += 1;
  }
  return count;
}

function hasComboTag(card, tag) {
  return Array.isArray(card?.comboTags) && card.comboTags.includes(tag);
}

function countCapturedComboTag(player, zone, tag) {
  const cards = player?.captured?.[zone] || [];
  if (!Array.isArray(cards)) return 0;
  const seen = new Set();
  let count = 0;
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    if (hasComboTag(card, tag)) count += 1;
  }
  return count;
}

function isDoublePiCard(card) {
  if (!card) return false;
  const id = String(card?.id || "");
  const category = String(card?.category || "");
  const piValue = Number(card?.piValue || 0);
  if (id === "I0") return true;
  return category === "junk" && piValue >= 2;
}

function resolveCandidateMonth(state, actor, decisionType, card) {
  const cardMonth = Number(card?.month || 0);
  if (cardMonth >= 1) return cardMonth;

  if (decisionType === "option") {
    if (state?.phase === "shaking-confirm" && state?.pendingShakingConfirm?.playerKey === actor) {
      const pendingMonth = Number(state?.pendingShakingConfirm?.month || 0);
      if (pendingMonth >= 1) return pendingMonth;
    }
    if (state?.phase === "president-choice" && state?.pendingPresident?.playerKey === actor) {
      const pendingMonth = Number(state?.pendingPresident?.month || 0);
      if (pendingMonth >= 1) return pendingMonth;
    }
  }
  return 0;
}

function matchOpportunityDensity(state, month) {
  const boardMonthCount = countCardsByMonth(state?.board || [], month);
  return clamp01(boardMonthCount / 3.0);
}

function immediateMatchPossible(state, decisionType, month) {
  if (decisionType === "match") return 1;
  if (month <= 0) return 0;
  return countCardsByMonth(state?.board || [], month) > 0 ? 1 : 0;
}

function monthTotalCards(month) {
  const m = Number(month || 0);
  if (m >= 1 && m <= 12) return 4;
  if (m === 13) return 2;
  return 0;
}

function collectKnownCardsForMonthRatio(state, actor) {
  const out = [];
  const pushAll = (cards) => {
    if (!Array.isArray(cards)) return;
    for (const card of cards) out.push(card);
  };

  pushAll(state?.board || []);
  for (const side of ["human", "ai"]) {
    const captured = state?.players?.[side]?.captured || {};
    pushAll(captured.kwang || []);
    pushAll(captured.five || []);
    pushAll(captured.ribbon || []);
    pushAll(captured.junk || []);
  }
  pushAll(state?.players?.[actor]?.hand || []);
  return out;
}

function candidateMonthKnownRatio(state, actor, month) {
  const total = monthTotalCards(month);
  if (total <= 0) return 0;

  const cards = collectKnownCardsForMonthRatio(state, actor);
  const seen = new Set();
  let known = 0;
  for (const card of cards) {
    if (!card) continue;
    const id = String(card.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    if (Number(card.month || 0) === Number(month)) known += 1;
  }
  return clamp01(known / total);
}

function decisionAvailabilityFlags(state, actor) {
  if (state?.phase === "shaking-confirm" && state?.pendingShakingConfirm?.playerKey === actor) {
    return { hasShake: 1, hasBomb: 0 };
  }
  if (state?.phase !== "playing" || state?.currentTurn !== actor) {
    return { hasShake: 0, hasBomb: 0 };
  }
  const shakingMonths = getDeclarableShakingMonths(state, actor);
  const bombMonths = getDeclarableBombMonths(state, actor);
  return {
    hasShake: Array.isArray(shakingMonths) && shakingMonths.length > 0 ? 1 : 0,
    hasBomb: Array.isArray(bombMonths) && bombMonths.length > 0 ? 1 : 0,
  };
}

function currentMultiplierNorm(state, scoreSelf) {
  const carry = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  const mul = Math.max(1.0, Number(scoreSelf?.multiplier || 1.0));
  const currentMultiplier = mul * carry;
  return clamp01((currentMultiplier - 1.0) / 15.0);
}

function featureVectorForCandidate(state, actor, decisionType, candidate, legalCount) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);

  const phase = String(state.phase || "");
  const card = candidateCard(state, actor, decisionType, candidate);
  const month = resolveCandidateMonth(state, actor, decisionType, card);
  const piValue = Number(card?.piValue || 0);
  const category = String(card?.category || "");
  const selfGwangCount = Number(state?.players?.[actor]?.captured?.kwang?.length || 0);
  const oppGwangCount = Number(state?.players?.[opp]?.captured?.kwang?.length || 0);
  const selfPiCount = Number(scoringPiCount(state.players[actor]) || 0);
  const oppPiCount = Number(scoringPiCount(state.players[opp]) || 0);

  const selfGodori = countCapturedComboTag(state.players?.[actor], "five", "fiveBirds");
  const oppGodori = countCapturedComboTag(state.players?.[opp], "five", "fiveBirds");
  const selfCheongdan = countCapturedComboTag(state.players?.[actor], "ribbon", "blueRibbons");
  const oppCheongdan = countCapturedComboTag(state.players?.[opp], "ribbon", "blueRibbons");
  const selfHongdan = countCapturedComboTag(state.players?.[actor], "ribbon", "redRibbons");
  const oppHongdan = countCapturedComboTag(state.players?.[opp], "ribbon", "redRibbons");
  const selfChodan = countCapturedComboTag(state.players?.[actor], "ribbon", "plainRibbons");
  const oppChodan = countCapturedComboTag(state.players?.[opp], "ribbon", "plainRibbons");

  const selfCanStop = Number(scoreSelf?.total || 0) >= 7 ? 1 : 0;
  const oppCanStop = Number(scoreOpp?.total || 0) >= 7 ? 1 : 0;
  const { hasShake, hasBomb } = decisionAvailabilityFlags(state, actor);

  return [
    phase === "playing" ? 1 : 0,
    phase === "select-match" ? 1 : 0,
    phase === "go-stop" ? 1 : 0,
    phase === "president-choice" ? 1 : 0,
    phase === "gukjin-choice" ? 1 : 0,
    phase === "shaking-confirm" ? 1 : 0,
    decisionType === "play" ? 1 : 0,
    decisionType === "match" ? 1 : 0,
    decisionType === "option" ? 1 : 0,
    clamp01((state.deck?.length || 0) / 30.0),
    clamp01((state.players?.[actor]?.hand?.length || 0) / 10.0),
    clamp01((state.players?.[opp]?.hand?.length || 0) / 10.0),
    clamp01((state.players?.[actor]?.goCount || 0) / 5.0),
    clamp01((state.players?.[opp]?.goCount || 0) / 5.0),
    tanhNorm((scoreSelf?.total || 0) - (scoreOpp?.total || 0), 10.0),
    tanhNorm((scoreSelf?.total || 0), 10.0),
    clamp01(Number(legalCount || 0) / 10.0),
    clamp01(piValue / 5.0),
    category === "kwang" ? 1 : 0,
    category === "ribbon" ? 1 : 0,
    category === "five" ? 1 : 0,
    category === "junk" ? 1 : 0,
    isDoublePiCard(card) ? 1 : 0,
    matchOpportunityDensity(state, month),
    immediateMatchPossible(state, decisionType, month),
    optionCode(candidate),
    clamp01(selfGwangCount / 5.0),
    clamp01(oppGwangCount / 5.0),
    clamp01(selfPiCount / 20.0),
    clamp01(oppPiCount / 20.0),
    clamp01(selfGodori / 3.0),
    clamp01(oppGodori / 3.0),
    clamp01(selfCheongdan / 3.0),
    clamp01(oppCheongdan / 3.0),
    clamp01(selfHongdan / 3.0),
    clamp01(oppHongdan / 3.0),
    clamp01(selfChodan / 3.0),
    clamp01(oppChodan / 3.0),
    selfCanStop,
    oppCanStop,
    hasShake,
    currentMultiplierNorm(state, scoreSelf),
    hasBomb,
    scoreSelf?.bak?.pi ? 1 : 0,
    scoreSelf?.bak?.gwang ? 1 : 0,
    scoreSelf?.bak?.mongBak ? 1 : 0,
    candidateMonthKnownRatio(state, actor, month),
  ];
}

function normalizeDecisionCandidate(decisionType, candidate) {
  if (decisionType === "option") return canonicalOptionAction(candidate);
  return String(candidate || "").trim();
}

function inferCandidateFromKiboTransition(stateBefore, actor, decisionType, candidates, stateAfter) {
  const beforeLen = Array.isArray(stateBefore?.kibo) ? stateBefore.kibo.length : 0;
  const afterKibo = Array.isArray(stateAfter?.kibo) ? stateAfter.kibo : [];
  if (afterKibo.length <= beforeLen) return null;

  const candidateSet = new Set((candidates || []).map((c) => normalizeDecisionCandidate(decisionType, c)));
  for (let i = afterKibo.length - 1; i >= beforeLen; i -= 1) {
    const ev = afterKibo[i];
    if (String(ev?.type || "") !== "turn_end") continue;
    if (String(ev?.actor || "") !== String(actor || "")) continue;
    const actionType = String(ev?.action?.type || "").trim().toLowerCase();

    if (decisionType === "play" && actionType === "play") {
      const playedId = String(ev?.action?.card?.id || "").trim();
      if (playedId && candidateSet.has(playedId)) return playedId;
    }
    if (decisionType === "match" && actionType === "play") {
      const selectedBoardId = String(ev?.action?.selectedBoardCard?.id || "").trim();
      if (selectedBoardId && candidateSet.has(selectedBoardId)) return selectedBoardId;
    }
    break;
  }
  return null;
}

function inferPlayCandidateFromHandDiff(stateBefore, actor, candidates, stateAfter) {
  const beforeHand = Array.isArray(stateBefore?.players?.[actor]?.hand) ? stateBefore.players[actor].hand : [];
  const afterHand = Array.isArray(stateAfter?.players?.[actor]?.hand) ? stateAfter.players[actor].hand : [];
  const beforeIds = new Set(beforeHand.map((c) => String(c?.id || "")).filter((id) => id.length > 0));
  const afterIds = new Set(afterHand.map((c) => String(c?.id || "")).filter((id) => id.length > 0));
  const removed = [];
  for (const candidate of candidates || []) {
    const id = String(candidate || "").trim();
    if (!id) continue;
    if (beforeIds.has(id) && !afterIds.has(id)) removed.push(id);
  }
  if (removed.length === 1) return removed[0];
  return null;
}

function inferChosenCandidateFromTransition(stateBefore, actor, decisionType, candidates, stateAfter) {
  if (!stateAfter || !Array.isArray(candidates) || !candidates.length) return null;

  // Prefer explicit action traces over simulation replay when available.
  const kiboInferred = inferCandidateFromKiboTransition(
    stateBefore,
    actor,
    decisionType,
    candidates,
    stateAfter
  );
  if (kiboInferred) return normalizeDecisionCandidate(decisionType, kiboInferred);

  // For play decisions, the selected card should disappear from actor hand.
  if (decisionType === "play") {
    const handDiffInferred = inferPlayCandidateFromHandDiff(stateBefore, actor, candidates, stateAfter);
    if (handDiffInferred) return normalizeDecisionCandidate(decisionType, handDiffInferred);
  }

  const target = stateProgressKey(stateAfter);
  for (const candidate of candidates) {
    const simulated = applyAction(stateBefore, actor, decisionType, candidate);
    if (simulated && stateProgressKey(simulated) === target) {
      return normalizeDecisionCandidate(decisionType, candidate);
    }
  }
  return null;
}

function randomChoice(arr, rng) {
  if (!arr.length) return null;
  const idx = Math.max(0, Math.min(arr.length - 1, Math.floor(Number(rng() || 0) * arr.length)));
  return arr[idx];
}

function randomLegalAction(state, actor, rng) {
  const sp = selectPool(state, actor);
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const options = sp.options || null;
  const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
  if (!decisionType) return state;
  const candidates = legalCandidatesForDecision(sp, decisionType);
  if (!candidates.length) return state;
  const picked = randomChoice(candidates, rng);
  return applyAction(state, actor, decisionType, picked);
}

function incrementCounter(map, key) {
  const k = String(key || "");
  if (!k) return;
  map[k] = Number(map[k] || 0) + 1;
}

function cardIdList(cards) {
  if (!Array.isArray(cards)) return [];
  return cards.map((c) => String(c?.id || "")).filter((id) => id.length > 0);
}

function tailKiboEvents(stateBefore, stateAfter) {
  const beforeLen = Array.isArray(stateBefore?.kibo) ? stateBefore.kibo.length : 0;
  const afterKibo = Array.isArray(stateAfter?.kibo) ? stateAfter.kibo : [];
  if (afterKibo.length <= beforeLen) return [];
  return afterKibo.slice(beforeLen);
}

function unresolvedTraceRow(decision, stateAfter, gameIndex, seed, firstTurnKey) {
  const actor = String(decision?.actor || "");
  const stateBefore = decision?.stateBefore || null;
  const beforePlayer = stateBefore?.players?.[actor] || {};
  const afterPlayer = stateAfter?.players?.[actor] || {};
  return {
    game_index: gameIndex,
    seed,
    first_turn: firstTurnKey,
    step: Number(decision?.step || 0),
    actor,
    actor_policy: String(decision?.policy || ""),
    action_source: String(decision?.actionSource || ""),
    decision_type: String(decision?.decisionType || ""),
    legal_count: Array.isArray(decision?.candidates) ? decision.candidates.length : 0,
    candidates: Array.isArray(decision?.candidates) ? decision.candidates : [],
    inferred_candidate: decision?.chosenCandidate || null,
    phase_before: String(stateBefore?.phase || ""),
    phase_after: String(stateAfter?.phase || ""),
    state_key_before: stateProgressKey(stateBefore),
    state_key_after: stateProgressKey(stateAfter),
    actor_hand_before: cardIdList(beforePlayer?.hand),
    actor_hand_after: cardIdList(afterPlayer?.hand),
    actor_capture_before: cardIdList(beforePlayer?.capture),
    actor_capture_after: cardIdList(afterPlayer?.capture),
    pending_match_before: stateBefore?.pendingMatch || null,
    pending_match_after: stateAfter?.pendingMatch || null,
    declarable_shaking_months: getDeclarableShakingMonths(stateBefore, actor),
    declarable_bomb_months: getDeclarableBombMonths(stateBefore, actor),
    kibo_delta: tailKiboEvents(stateBefore, stateAfter),
  };
}

function resolveFirstTurnKey(opts, gameIndex) {
  if (opts.firstTurnPolicy === "fixed") return opts.fixedFirstTurn;
  return gameIndex % 2 === 0 ? "ai" : "human";
}

function startRound(seed, firstTurnKey, kiboDetail) {
  return initSimulationGame("A", createSeededRng(`${seed}|game`), {
    kiboDetail,
    firstTurnKey,
  });
}

function continueRound(prevEndState, seed, firstTurnKey, kiboDetail) {
  return startSimulationGame(prevEndState, createSeededRng(`${seed}|game`), {
    kiboDetail,
    keepGold: true,
    useCarryOver: true,
    firstTurnKey,
  });
}

function goldDiffByActor(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfGold = Number(state?.players?.[actor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return selfGold - oppGold;
}

function playSingleRound(initialState, seed, policyByActor, maxSteps, onDecision = null) {
  const rng = createSeededRng(`${seed}|rng`);
  let state = initialState;
  let steps = 0;

  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = stateProgressKey(state);
    const sp = selectPool(state, actor);
    const cards = sp.cards || null;
    const boardCardIds = sp.boardCardIds || null;
    const options = sp.options || null;
    const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
    const candidates = decisionType ? legalCandidatesForDecision(sp, decisionType) : [];
    const policy = policyByActor[actor];
    let actionSource = "heuristic";
    let next = aiPlay(state, actor, {
      source: "heuristic",
      heuristicPolicy: policy,
    });

    if (!next || stateProgressKey(next) === before) {
      actionSource = "fallback_random";
      next = randomLegalAction(state, actor, rng);
    }
    if (!next || stateProgressKey(next) === before) {
      break;
    }

    if (typeof onDecision === "function" && decisionType && candidates.length > 0) {
      const chosenCandidate = inferChosenCandidateFromTransition(
        state,
        actor,
        decisionType,
        candidates,
        next
      );
      onDecision({
        stateBefore: state,
        stateAfter: next,
        actor,
        policy,
        decisionType,
        candidates,
        chosenCandidate,
        actionSource,
        step: steps,
      });
    }

    state = next;
    steps += 1;
  }

  return state;
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function createSeatRecord() {
  return {
    games: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    go_count_total: 0,
    go_game_count: 0,
    go_fail_count: 0,
    gold_deltas: [],
  };
}

function updateSeatRecord(record, winner, selfActor, oppActor, goldDelta, selfGoCount) {
  record.games += 1;
  if (winner === selfActor) record.wins += 1;
  else if (winner === oppActor) record.losses += 1;
  else record.draws += 1;
  const goCount = Math.max(0, Number(selfGoCount || 0));
  record.go_count_total += goCount;
  if (goCount > 0) {
    record.go_game_count += 1;
    if (winner !== selfActor) {
      record.go_fail_count += 1;
    }
  }
  record.gold_deltas.push(Number(goldDelta || 0));
}

function finalizeSeatRecord(record) {
  const games = Number(record?.games || 0);
  const wins = Number(record?.wins || 0);
  const losses = Number(record?.losses || 0);
  const draws = Number(record?.draws || 0);
  const goCountTotal = Number(record?.go_count_total || 0);
  const goGameCount = Number(record?.go_game_count || 0);
  const goFailCount = Number(record?.go_fail_count || 0);
  const deltas = Array.isArray(record?.gold_deltas) ? record.gold_deltas : [];
  const meanGoldDelta = deltas.length > 0 ? deltas.reduce((a, b) => a + b, 0) / deltas.length : 0;
  return {
    games,
    wins,
    losses,
    draws,
    win_rate: games > 0 ? wins / games : 0,
    loss_rate: games > 0 ? losses / games : 0,
    draw_rate: games > 0 ? draws / games : 0,
    go_count_total: goCountTotal,
    go_avg_per_game: games > 0 ? goCountTotal / games : 0,
    go_game_count: goGameCount,
    go_fail_count: goFailCount,
    go_success_count: Math.max(0, goGameCount - goFailCount),
    go_fail_rate: goGameCount > 0 ? goFailCount / goGameCount : 0,
    mean_gold_delta: meanGoldDelta,
    p10_gold_delta: quantile(deltas, 0.1),
    p50_gold_delta: quantile(deltas, 0.5),
    p90_gold_delta: quantile(deltas, 0.9),
  };
}

function buildSeatSplitSummary(firstRecord, secondRecord) {
  const combined = createSeatRecord();
  combined.games = Number(firstRecord.games || 0) + Number(secondRecord.games || 0);
  combined.wins = Number(firstRecord.wins || 0) + Number(secondRecord.wins || 0);
  combined.losses = Number(firstRecord.losses || 0) + Number(secondRecord.losses || 0);
  combined.draws = Number(firstRecord.draws || 0) + Number(secondRecord.draws || 0);
  combined.go_count_total =
    Number(firstRecord.go_count_total || 0) + Number(secondRecord.go_count_total || 0);
  combined.go_game_count =
    Number(firstRecord.go_game_count || 0) + Number(secondRecord.go_game_count || 0);
  combined.go_fail_count =
    Number(firstRecord.go_fail_count || 0) + Number(secondRecord.go_fail_count || 0);
  combined.gold_deltas = [
    ...(Array.isArray(firstRecord.gold_deltas) ? firstRecord.gold_deltas : []),
    ...(Array.isArray(secondRecord.gold_deltas) ? secondRecord.gold_deltas : []),
  ];
  return {
    when_first: finalizeSeatRecord(firstRecord),
    when_second: finalizeSeatRecord(secondRecord),
    combined: finalizeSeatRecord(combined),
  };
}

function main() {
  const evalStartMs = Date.now();
  const opts = parseArgs(process.argv.slice(2));
  let kiboWriter = null;
  if (opts.kiboOut) {
    mkdirSync(dirname(opts.kiboOut), { recursive: true });
    kiboWriter = createWriteStream(opts.kiboOut, { flags: "w", encoding: "utf8" });
  }
  let datasetWriter = null;
  const datasetStats = {
    rows: 0,
    positive_rows: 0,
    decisions: 0,
    unresolved_decisions: 0,
  };
  let unresolvedWriter = null;
  const unresolvedStats = {
    decisions: 0,
    unresolved_decisions: 0,
    unresolved_rows: 0,
    by_actor: {},
    by_policy: {},
    by_decision_type: {},
    by_action_source: {},
  };
  if (opts.datasetOut) {
    mkdirSync(dirname(opts.datasetOut), { recursive: true });
    datasetWriter = createWriteStream(opts.datasetOut, { flags: "w", encoding: "utf8" });
  }
  if (opts.unresolvedOut) {
    mkdirSync(dirname(opts.unresolvedOut), { recursive: true });
    unresolvedWriter = createWriteStream(opts.unresolvedOut, { flags: "w", encoding: "utf8" });
  }

  const actorA = "human";
  const actorB = "ai";
  const policyByActor = {
    [actorA]: opts.policyA,
    [actorB]: opts.policyB,
  };

  let winsA = 0;
  let winsB = 0;
  let draws = 0;
  const goldDeltasA = [];
  const bankrupt = {
    a_bankrupt_count: 0,
    b_bankrupt_count: 0,
  };
  const firstTurnCounts = {
    human: 0,
    ai: 0,
  };
  const seatSplitA = {
    first: createSeatRecord(),
    second: createSeatRecord(),
  };
  const seatSplitB = {
    first: createSeatRecord(),
    second: createSeatRecord(),
  };
  const seriesSession = {
    roundsPlayed: 0,
    previousEndState: null,
  };

  for (let gi = 0; gi < opts.games; gi += 1) {
    const firstTurnKey = resolveFirstTurnKey(opts, gi);
    firstTurnCounts[firstTurnKey] += 1;
    const seed = `${opts.seed}|g=${gi}|first=${firstTurnKey}|sr=${seriesSession.roundsPlayed}`;

    const roundStart = opts.continuousSeries
      ? seriesSession.previousEndState
        ? continueRound(seriesSession.previousEndState, seed, firstTurnKey, opts.kiboDetail)
        : startRound(seed, firstTurnKey, opts.kiboDetail)
      : startRound(seed, firstTurnKey, opts.kiboDetail);

    const beforeDiffA = goldDiffByActor(roundStart, actorA);
    const endState = playSingleRound(
      roundStart,
      seed,
      policyByActor,
      Math.max(20, Math.floor(opts.maxSteps)),
      (decision) => {
        if (!datasetWriter && !unresolvedWriter) return;
        if (opts.datasetActor !== "all" && decision.actor !== opts.datasetActor) return;
        if (datasetWriter) datasetStats.decisions += 1;
        unresolvedStats.decisions += 1;
        const legalCount = decision.candidates.length;
        if (legalCount <= 0) return;
        let matched = false;
        if (datasetWriter) {
          for (const candidate of decision.candidates) {
            const candidateNorm = normalizeDecisionCandidate(decision.decisionType, candidate);
            const isChosen = candidateNorm === decision.chosenCandidate ? 1 : 0;
            if (isChosen) matched = true;
            const row = {
              game_index: gi,
              seed,
              first_turn: firstTurnKey,
              step: decision.step,
              actor: decision.actor,
              actor_policy: decision.policy,
              action_source: decision.actionSource,
              decision_type: decision.decisionType,
              legal_count: legalCount,
              candidate: candidateNorm,
              chosen: isChosen,
              chosen_candidate: decision.chosenCandidate,
              features: featureVectorForCandidate(
                decision.stateBefore,
                decision.actor,
                decision.decisionType,
                candidate,
                legalCount
              ),
            };
            datasetWriter.write(`${JSON.stringify(row)}\n`);
            datasetStats.rows += 1;
            if (isChosen) datasetStats.positive_rows += 1;
          }
        } else {
          for (const candidate of decision.candidates) {
            const candidateNorm = normalizeDecisionCandidate(decision.decisionType, candidate);
            if (candidateNorm === decision.chosenCandidate) {
              matched = true;
              break;
            }
          }
        }
        if (!matched) {
          if (datasetWriter) datasetStats.unresolved_decisions += 1;
          unresolvedStats.unresolved_decisions += 1;
          incrementCounter(unresolvedStats.by_actor, decision.actor);
          incrementCounter(unresolvedStats.by_policy, decision.policy);
          incrementCounter(unresolvedStats.by_decision_type, decision.decisionType);
          incrementCounter(unresolvedStats.by_action_source, decision.actionSource);
          const withinLimit =
            opts.unresolvedLimit <= 0 || unresolvedStats.unresolved_rows < opts.unresolvedLimit;
          if (withinLimit && unresolvedWriter) {
            const traceRow = unresolvedTraceRow(decision, decision.stateAfter, gi, seed, firstTurnKey);
            unresolvedWriter.write(`${JSON.stringify(traceRow)}\n`);
            unresolvedStats.unresolved_rows += 1;
          }
        }
      }
    );
    const afterDiffA = goldDiffByActor(endState, actorA);
    goldDeltasA.push(afterDiffA - beforeDiffA);

    const goldA = Number(endState?.players?.[actorA]?.gold || 0);
    const goldB = Number(endState?.players?.[actorB]?.gold || 0);
    const goCountA = Math.max(0, Number(endState?.players?.[actorA]?.goCount || 0));
    const goCountB = Math.max(0, Number(endState?.players?.[actorB]?.goCount || 0));
    if (goldA <= 0) bankrupt.a_bankrupt_count += 1;
    if (goldB <= 0) bankrupt.b_bankrupt_count += 1;

    if (opts.continuousSeries) {
      seriesSession.previousEndState = endState;
    }
    seriesSession.roundsPlayed += 1;

    const winner = String(endState?.result?.winner || "").trim();
    if (winner === actorA) winsA += 1;
    else if (winner === actorB) winsB += 1;
    else draws += 1;

    if (kiboWriter) {
      const kiboRecord = {
        game_index: gi,
        seed,
        first_turn: firstTurnKey,
        policy_a: opts.policyA,
        policy_b: opts.policyB,
        winner,
        result: endState?.result || null,
        kibo_detail: endState?.kiboDetail || opts.kiboDetail,
        kibo: Array.isArray(endState?.kibo) ? endState.kibo : [],
      };
      kiboWriter.write(`${JSON.stringify(kiboRecord)}\n`);
    }

    const seatAKey = firstTurnKey === actorA ? "first" : "second";
    const seatBKey = firstTurnKey === actorB ? "first" : "second";
    const goldDeltaA = afterDiffA - beforeDiffA;
    updateSeatRecord(seatSplitA[seatAKey], winner, actorA, actorB, goldDeltaA, goCountA);
    updateSeatRecord(seatSplitB[seatBKey], winner, actorB, actorA, -goldDeltaA, goCountB);
  }

  const games = opts.games;
  const winRateA = winsA / games;
  const winRateB = winsB / games;
  const drawRate = draws / games;
  const meanGoldDeltaA =
    goldDeltasA.length > 0 ? goldDeltasA.reduce((a, b) => a + b, 0) / goldDeltasA.length : 0;
  const lossesA = winsB;
  const lossesB = winsA;
  const splitSummaryA = buildSeatSplitSummary(seatSplitA.first, seatSplitA.second);
  const splitSummaryB = buildSeatSplitSummary(seatSplitB.first, seatSplitB.second);

  const summary = {
    games,
    actor_a: actorA,
    actor_b: actorB,
    policy_a: opts.policyA,
    policy_b: opts.policyB,
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
    kibo_detail: opts.kiboDetail,
    kibo_out: opts.kiboOut || null,
    dataset_out: opts.datasetOut || null,
    dataset_actor: opts.datasetActor,
    dataset_rows: datasetStats.rows,
    dataset_positive_rows: datasetStats.positive_rows,
    dataset_decisions: datasetStats.decisions,
    dataset_unresolved_decisions: datasetStats.unresolved_decisions,
    unresolved_out: opts.unresolvedOut || null,
    unresolved_limit: opts.unresolvedLimit,
    unresolved_rows: unresolvedStats.unresolved_rows,
    unresolved_decisions: unresolvedStats.unresolved_decisions,
    unresolved_decision_rate:
      unresolvedStats.decisions > 0 ? unresolvedStats.unresolved_decisions / unresolvedStats.decisions : 0,
    unresolved_by_actor: unresolvedStats.by_actor,
    unresolved_by_policy: unresolvedStats.by_policy,
    unresolved_by_decision_type: unresolvedStats.by_decision_type,
    unresolved_by_action_source: unresolvedStats.by_action_source,
    bankrupt,
    session_rounds: {
      total_rounds: seriesSession.roundsPlayed,
    },
    wins_a: winsA,
    losses_a: lossesA,
    wins_b: winsB,
    losses_b: lossesB,
    draws,
    win_rate_a: winRateA,
    win_rate_b: winRateB,
    draw_rate: drawRate,
    go_count_a: splitSummaryA.combined.go_count_total,
    go_count_b: splitSummaryB.combined.go_count_total,
    go_games_a: splitSummaryA.combined.go_game_count,
    go_games_b: splitSummaryB.combined.go_game_count,
    go_fail_count_a: splitSummaryA.combined.go_fail_count,
    go_fail_count_b: splitSummaryB.combined.go_fail_count,
    go_fail_rate_a: splitSummaryA.combined.go_fail_rate,
    go_fail_rate_b: splitSummaryB.combined.go_fail_rate,
    mean_gold_delta_a: meanGoldDeltaA,
    p10_gold_delta_a: quantile(goldDeltasA, 0.1),
    p50_gold_delta_a: quantile(goldDeltasA, 0.5),
    p90_gold_delta_a: quantile(goldDeltasA, 0.9),
    seat_split_a: splitSummaryA,
    seat_split_b: splitSummaryB,
    eval_time_ms: Math.max(0, Date.now() - evalStartMs),
  };

  if (kiboWriter) kiboWriter.end();
  if (datasetWriter) datasetWriter.end();
  if (unresolvedWriter) unresolvedWriter.end();

  process.stdout.write(`${JSON.stringify(summary)}\n`);
}

try {
  main();
} catch (err) {
  const msg = err && err.stack ? err.stack : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
}
