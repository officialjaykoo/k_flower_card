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

function normalizeLearningRole(value) {
  const v = String(value || "").trim().toUpperCase();
  if (v === "ATTACK" || v === "DEFENSE") return v;
  return "NEUTRAL";
}

function roleForSideFromLearningRole(learningRole, side) {
  const lr = normalizeLearningRole(learningRole);
  // learningRole is defined from mySide perspective.
  if (side === "mySide") return lr;
  if (lr === "ATTACK") return "DEFENSE";
  if (lr === "DEFENSE") return "ATTACK";
  return "NEUTRAL";
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
    droppedByMinTraces: 0,
    droppedByNeutralRole: 0
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

        const gameLearningRole = normalizeLearningRole(game?.learningRole);
        const actorRole = roleForSideFromLearningRole(gameLearningRole, side);
        if (actorRole !== "ATTACK" && actorRole !== "DEFENSE") {
          stats.droppedByNeutralRole += 1;
          continue;
        }

        if (actorRole === "ATTACK") {
          const out = {
            ...game,
            decision_trace: actorTraces,
            label: {
              role: "ATTACK",
              roleSource: "learningRole",
              gameLearningRole,
              actorTag: cfg.actorTag,
              actorSide: side,
              actorPolicy: sidePolicy(game, side),
              sourceFile: p
            }
          };
          aw.write(`${JSON.stringify(out)}\n`);
          stats.attackGames += 1;
          stats.attackTraces += actorTraces.length;
        }

        if (actorRole === "DEFENSE") {
          const out = {
            ...game,
            decision_trace: actorTraces,
            label: {
              role: "DEFENSE",
              roleSource: "learningRole",
              gameLearningRole,
              actorTag: cfg.actorTag,
              actorSide: side,
              actorPolicy: sidePolicy(game, side),
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

  const stats = await writeLabeled(
    {
      attackOut: args.attackOut,
      defenseOut: args.defenseOut
    },
    {
      inputs: args.inputs,
      actorTag: args.actorTag,
      minActorTraces: args.minActorTraces
    }
  );

  if ((stats.attackGames + stats.defenseGames) <= 0) {
    throw new Error(
      "No ATTACK/DEFENSE labels found from learningRole. Input logs likely do not contain learningRole (or are all NEUTRAL)."
    );
  }

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
          attack: "learningRole == ATTACK (actor perspective)",
          defense: "learningRole == DEFENSE (actor perspective)"
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
