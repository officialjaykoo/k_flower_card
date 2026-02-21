#!/usr/bin/env node
import fs from "fs";
import path from "path";
import readline from "readline";

function parseArgs(argv) {
  const args = {
    inputs: [],
    attackOut: "logs/train_delta_v4_label_attack.jsonl",
    defenseOut: "logs/train_delta_v4_label_defense.jsonl",
    actorTag: "heuristic_v4",
    attackTopPercent: 20,
    defenseOppScoreMax: 9,
    minActorTraces: 1
  };

  const rest = [...argv];
  while (rest.length) {
    const arg = String(rest.shift() || "");
    if (arg === "--input" && rest.length) {
      args.inputs.push(String(rest.shift()));
      continue;
    }
    if (arg.startsWith("--input=")) {
      args.inputs.push(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--attack-out" && rest.length) {
      args.attackOut = String(rest.shift());
      continue;
    }
    if (arg.startsWith("--attack-out=")) {
      args.attackOut = arg.split("=", 2)[1];
      continue;
    }
    if (arg === "--defense-out" && rest.length) {
      args.defenseOut = String(rest.shift());
      continue;
    }
    if (arg.startsWith("--defense-out=")) {
      args.defenseOut = arg.split("=", 2)[1];
      continue;
    }
    if (arg === "--actor-tag" && rest.length) {
      args.actorTag = String(rest.shift());
      continue;
    }
    if (arg.startsWith("--actor-tag=")) {
      args.actorTag = arg.split("=", 2)[1];
      continue;
    }
    if (arg === "--attack-top-percent" && rest.length) {
      args.attackTopPercent = Number(rest.shift());
      continue;
    }
    if (arg.startsWith("--attack-top-percent=")) {
      args.attackTopPercent = Number(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--defense-opp-score-max" && rest.length) {
      args.defenseOppScoreMax = Number(rest.shift());
      continue;
    }
    if (arg.startsWith("--defense-opp-score-max=")) {
      args.defenseOppScoreMax = Number(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--min-actor-traces" && rest.length) {
      args.minActorTraces = Number(rest.shift());
      continue;
    }
    if (arg.startsWith("--min-actor-traces=")) {
      args.minActorTraces = Number(arg.split("=", 2)[1]);
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!args.inputs.length) {
    throw new Error("Missing --input <path>. Provide one or more --input.");
  }
  if (!Number.isFinite(args.attackTopPercent) || args.attackTopPercent <= 0 || args.attackTopPercent > 100) {
    throw new Error(`Invalid --attack-top-percent: ${args.attackTopPercent}`);
  }
  if (!Number.isFinite(args.defenseOppScoreMax)) {
    throw new Error(`Invalid --defense-opp-score-max: ${args.defenseOppScoreMax}`);
  }
  if (!Number.isFinite(args.minActorTraces) || args.minActorTraces < 1) {
    throw new Error(`Invalid --min-actor-traces: ${args.minActorTraces}`);
  }
  return args;
}

function assertFilesExist(paths) {
  const missing = paths.filter((p) => !fs.existsSync(p));
  if (missing.length) {
    throw new Error(`Missing input files:\n${missing.map((x) => `- ${x}`).join("\n")}`);
  }
}

function sidePolicy(game, side) {
  const run = game?.run?.[side];
  if (run != null) return String(run);
  const pol = game?.policy?.[side];
  if (pol != null) return String(pol);
  return "";
}

function sideMatchesActor(game, side, actorTagLower) {
  const policy = sidePolicy(game, side).toLowerCase();
  return policy.includes(actorTagLower);
}

function sideGoldDelta(game, side) {
  const dMy = Number(game?.goldDeltaMy);
  if (Number.isFinite(dMy)) return side === "mySide" ? dMy : -dMy;

  const fMy = Number(game?.finalGoldMy);
  const iMy = Number(game?.initialGoldMy);
  if (Number.isFinite(fMy) && Number.isFinite(iMy)) {
    const deltaMy = fMy - iMy;
    return side === "mySide" ? deltaMy : -deltaMy;
  }

  const fYour = Number(game?.finalGoldYour);
  const iYour = Number(game?.initialGoldYour);
  if (Number.isFinite(fYour) && Number.isFinite(iYour)) {
    const deltaYour = fYour - iYour;
    return side === "yourSide" ? deltaYour : -deltaYour;
  }
  return 0;
}

function sideScore(game, side) {
  const s = Number(game?.score?.[side]);
  if (Number.isFinite(s)) return s;
  return 0;
}

function otherSide(side) {
  return side === "mySide" ? "yourSide" : "mySide";
}

function isWinner(game, side) {
  return String(game?.winner || "") === side;
}

function isLoser(game, side) {
  const w = String(game?.winner || "");
  if (w === "draw" || w === "unknown" || !w) return false;
  return w !== side;
}

function quantileThreshold(sortedValues, q01) {
  if (!sortedValues.length) return null;
  const q = Math.max(0, Math.min(1, q01));
  const idx = Math.max(0, Math.min(sortedValues.length - 1, Math.floor((sortedValues.length - 1) * q)));
  return sortedValues[idx];
}

async function collectAttackThreshold(inputs, actorTag, attackTopPercent) {
  const actorTagLower = String(actorTag || "").toLowerCase();
  const winningGold = [];
  let gamesRead = 0;

  for (const p of inputs) {
    const rl = readline.createInterface({
      input: fs.createReadStream(p, { encoding: "utf8" }),
      crlfDelay: Infinity
    });
    for await (const raw of rl) {
      const line = String(raw || "").trim();
      if (!line) continue;
      gamesRead += 1;
      let game;
      try {
        game = JSON.parse(line);
      } catch (err) {
        throw new Error(`JSON parse failed (${p}, line ${gamesRead}): ${String(err)}`);
      }
      for (const side of ["mySide", "yourSide"]) {
        if (!sideMatchesActor(game, side, actorTagLower)) continue;
        if (isWinner(game, side)) {
          winningGold.push(sideGoldDelta(game, side));
        }
      }
    }
  }

  winningGold.sort((a, b) => a - b);
  const q = 1 - attackTopPercent / 100;
  const threshold = quantileThreshold(winningGold, q);
  return {
    threshold,
    winningCount: winningGold.length
  };
}

function filteredActorTraces(game, side) {
  const traces = Array.isArray(game?.decision_trace) ? game.decision_trace : [];
  return traces.filter((t) => t && String(t.a || "") === side);
}

async function writeLabeled(outputs, cfg) {
  fs.mkdirSync(path.dirname(outputs.attackOut) || ".", { recursive: true });
  fs.mkdirSync(path.dirname(outputs.defenseOut) || ".", { recursive: true });
  const aw = fs.createWriteStream(outputs.attackOut, { encoding: "utf8" });
  const dw = fs.createWriteStream(outputs.defenseOut, { encoding: "utf8" });

  const actorTagLower = String(cfg.actorTag || "").toLowerCase();
  const stats = {
    gamesRead: 0,
    actorSidesSeen: 0,
    attackGames: 0,
    defenseGames: 0,
    attackTraces: 0,
    defenseTraces: 0,
    droppedByMinTraces: 0
  };

  for (const p of cfg.inputs) {
    const rl = readline.createInterface({
      input: fs.createReadStream(p, { encoding: "utf8" }),
      crlfDelay: Infinity
    });
    for await (const raw of rl) {
      const line = String(raw || "").trim();
      if (!line) continue;
      stats.gamesRead += 1;
      let game;
      try {
        game = JSON.parse(line);
      } catch (err) {
        throw new Error(`JSON parse failed (${p}, line ${stats.gamesRead}): ${String(err)}`);
      }

      for (const side of ["mySide", "yourSide"]) {
        if (!sideMatchesActor(game, side, actorTagLower)) continue;
        stats.actorSidesSeen += 1;

        const actorTraces = filteredActorTraces(game, side);
        if (actorTraces.length < cfg.minActorTraces) {
          stats.droppedByMinTraces += 1;
          continue;
        }

        const opp = otherSide(side);
        const actorDelta = sideGoldDelta(game, side);
        const oppScore = sideScore(game, opp);
        const actorWin = isWinner(game, side);
        const actorLose = isLoser(game, side);

        const attackPass =
          actorWin &&
          cfg.attackThreshold != null &&
          actorDelta >= cfg.attackThreshold;
        const defensePass =
          actorLose &&
          oppScore <= cfg.defenseOppScoreMax;

        if (attackPass) {
          const out = {
            ...game,
            decision_trace: actorTraces,
            label: {
              role: "attack",
              actorTag: cfg.actorTag,
              actorSide: side,
              actorPolicy: sidePolicy(game, side),
              actorGoldDelta: actorDelta,
              attackTopPercent: cfg.attackTopPercent,
              attackThreshold: cfg.attackThreshold,
              sourceFile: p
            }
          };
          aw.write(`${JSON.stringify(out)}\n`);
          stats.attackGames += 1;
          stats.attackTraces += actorTraces.length;
        }

        if (defensePass) {
          const out = {
            ...game,
            decision_trace: actorTraces,
            label: {
              role: "defense",
              actorTag: cfg.actorTag,
              actorSide: side,
              actorPolicy: sidePolicy(game, side),
              actorGoldDelta: actorDelta,
              opponentScore: oppScore,
              defenseOppScoreMax: cfg.defenseOppScoreMax,
              sourceFile: p
            }
          };
          dw.write(`${JSON.stringify(out)}\n`);
          stats.defenseGames += 1;
          stats.defenseTraces += actorTraces.length;
        }
      }
    }
  }

  await new Promise((resolve) => aw.end(resolve));
  await new Promise((resolve) => dw.end(resolve));
  return stats;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  assertFilesExist(args.inputs);

  const thresholdInfo = await collectAttackThreshold(args.inputs, args.actorTag, args.attackTopPercent);
  const attackThreshold = thresholdInfo.threshold;
  if (attackThreshold == null) {
    throw new Error("No winning samples found for actor tag. Cannot compute attack top-percent threshold.");
  }

  const stats = await writeLabeled(
    {
      attackOut: args.attackOut,
      defenseOut: args.defenseOut
    },
    {
      inputs: args.inputs,
      actorTag: args.actorTag,
      attackTopPercent: args.attackTopPercent,
      attackThreshold,
      defenseOppScoreMax: args.defenseOppScoreMax,
      minActorTraces: args.minActorTraces
    }
  );

  console.log(
    JSON.stringify(
      {
        actorTag: args.actorTag,
        inputs: args.inputs,
        outputs: {
          attack: args.attackOut,
          defense: args.defenseOut
        },
        criteria: {
          attack: `actor win + top ${args.attackTopPercent}% by actor gold delta`,
          defense: `actor loss + opponent score <= ${args.defenseOppScoreMax}`
        },
        threshold: {
          attackGoldDeltaMin: attackThreshold,
          winningSamplesForThreshold: thresholdInfo.winningCount
        },
        stats
      },
      null,
      2
    )
  );
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});
