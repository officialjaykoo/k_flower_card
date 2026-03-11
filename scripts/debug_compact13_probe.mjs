import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { initSimulationGame, createSeededRng } from "../src/engine/index.js";
import { getActionPlayerKey } from "../src/engine/runner.js";
import { aiPlay } from "../src/ai/aiPlay.js";
import { debugFeatureRows } from "../src/ai/modelPolicyEngine.js";

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    genome: "",
    seed: "compact13-probe",
    firstTurn: "ai",
    maxSteps: 80,
    actor: "ai",
    inputDim: 13,
    opponentPolicy: "H-CL",
  };
  while (args.length > 0) {
    const key = String(args.shift() || "");
    const value = String(args.shift() || "");
    if (key === "--genome") out.genome = value;
    else if (key === "--seed") out.seed = value || out.seed;
    else if (key === "--first-turn") out.firstTurn = value || out.firstTurn;
    else if (key === "--max-steps") out.maxSteps = Math.max(1, Number(value || out.maxSteps));
    else if (key === "--actor") out.actor = value || out.actor;
    else if (key === "--input-dim") out.inputDim = Math.max(1, Number(value || out.inputDim));
    else if (key === "--opponent-policy") out.opponentPolicy = value || out.opponentPolicy;
    else throw new Error(`Unknown argument: ${key}`);
  }
  if (!out.genome) {
    throw new Error("--genome is required");
  }
  return out;
}

function loadGenomeModel(rawPath) {
  const fullPath = resolve(String(rawPath || "").trim());
  return JSON.parse(String(readFileSync(fullPath, "utf8") || "").replace(/^\uFEFF/, ""));
}

function transitionKey(state) {
  if (!state) return "null";
  return [
    String(state.phase || ""),
    String(state.currentTurn || ""),
    String(state.pendingGoStop || ""),
    String(state.pendingMatch?.stage || ""),
    String(state.turnSeq || 0),
    String(state.kiboSeq || 0),
  ].join("|");
}

function printDecisionProbe(label, probe) {
  if (!probe) {
    console.log(`${label}: no legal decision`);
    return;
  }
  console.log(`${label}: actor=${probe.actor} type=${probe.decisionType} inputDim=${probe.inputDim} legal=${probe.legalCount}`);
  for (const row of probe.rows) {
    console.log(
      JSON.stringify(
        {
          candidate: row.candidate,
          hasNaN: row.hasNaN,
          hasInfinity: row.hasInfinity,
          minValue: row.minValue,
          maxValue: row.maxValue,
          score: row.score,
          features: row.features,
        },
        null,
        2
      )
    );
  }
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const model = loadGenomeModel(opts.genome);
  let state = initSimulationGame("A", createSeededRng(`${opts.seed}|game`), {
    kiboDetail: "full",
    firstTurnKey: String(opts.firstTurn || "ai"),
  });

  let steps = 0;
  while (state.phase !== "resolution" && steps < Number(opts.maxSteps || 80)) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;
    const before = transitionKey(state);
    if (actor === opts.actor) {
      const probe = debugFeatureRows(state, actor, { policyModel: model, inputDim: opts.inputDim });
      printDecisionProbe(`step=${steps}`, probe);
      state = aiPlay(state, actor, { source: "model", model });
    } else {
      state = aiPlay(state, actor, { source: "heuristic", heuristicPolicy: opts.opponentPolicy });
    }
    const after = transitionKey(state);
    if (after === before) {
      throw new Error(`state did not advance at step=${steps}, actor=${actor}`);
    }
    steps += 1;
  }
  console.log(
    JSON.stringify(
      {
        finished: state.phase === "resolution",
        steps,
        result: state.result || null,
      },
      null,
      2
    )
  );
}

main();
