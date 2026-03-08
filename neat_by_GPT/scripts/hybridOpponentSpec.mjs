import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { resolveBotPolicy } from "../../src/ai/policies.js";

const SCRIPT_FILE = fileURLToPath(import.meta.url);
const SCRIPT_DIR = dirname(SCRIPT_FILE);
const REPO_ROOT = resolve(SCRIPT_DIR, "..", "..");

function sanitizeFilePart(text) {
  return String(text || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function toAbsoluteRepoPath(rawPath) {
  const token = String(rawPath || "").trim();
  if (!token) return "";
  return resolve(REPO_ROOT, token);
}

export function parseHybridPlayGoSpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_play_go\(\s*([^,]+)\s*,\s*([^,]+?)(?:\s*,\s*([^)]+)\s*)?\)$/i);
  if (!m) return null;
  return {
    modelToken: String(m[1] || "").trim(),
    goStopToken: String(m[2] || "").trim(),
    heuristicToken: String(m[3] || "").trim(),
  };
}

export function parseHybridPlaySpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_play\(\s*([^,]+)\s*,\s*([^)]+)\s*\)$/i);
  if (!m) return null;
  return {
    modelToken: String(m[1] || "").trim(),
    heuristicToken: String(m[2] || "").trim(),
  };
}

function tryResolveMainModelArtifactPath(rawSpec) {
  const token = String(rawSpec || "").trim();
  if (!token) return null;

  const phaseMatch = token.match(/^phase([0-3])_seed(\d+)$/i);
  if (phaseMatch) {
    const phase = Number(phaseMatch[1]);
    const seed = Number(phaseMatch[2]);
    const modelPath = join(REPO_ROOT, "logs", "NEAT", `neat_phase${phase}_seed${seed}`, "models", "winner_genome.json");
    return {
      path: modelPath,
      source: "named_main_run",
      exists: existsSync(modelPath),
      label: `phase${phase}_seed${seed}`,
      phase,
      seed,
    };
  }

  const explicitPath = token.startsWith("model:") ? token.slice("model:".length).trim() : token;
  const looksLikePath =
    token.startsWith("model:") ||
    explicitPath.includes("/") ||
    explicitPath.includes("\\") ||
    explicitPath.toLowerCase().endsWith(".json");
  if (!looksLikePath) return null;

  const absolute = toAbsoluteRepoPath(explicitPath);
  if (!existsSync(absolute)) {
    return {
      path: absolute,
      source: "explicit_path",
      exists: false,
      label: sanitizeFilePart(token).replace(/^model_/, "") || "main_model",
      phase: null,
      seed: null,
    };
  }

  const stat = statSync(absolute);
  if (stat.isDirectory()) {
    const dirCandidate = join(absolute, "models", "winner_genome.json");
    const fileCandidate = join(absolute, "winner_genome.json");
    if (existsSync(dirCandidate)) {
      return {
        path: dirCandidate,
        source: "explicit_run_dir",
        exists: true,
        label: sanitizeFilePart(token).replace(/^model_/, "") || "main_model",
        phase: null,
        seed: null,
      };
    }
    if (existsSync(fileCandidate)) {
      return {
        path: fileCandidate,
        source: "explicit_dir_file",
        exists: true,
        label: sanitizeFilePart(token).replace(/^model_/, "") || "main_model",
        phase: null,
        seed: null,
      };
    }
    return {
      path: dirCandidate,
      source: "explicit_run_dir",
      exists: false,
      label: sanitizeFilePart(token).replace(/^model_/, "") || "main_model",
      phase: null,
      seed: null,
    };
  }

  return {
    path: absolute,
    source: "explicit_file",
    exists: true,
    label: sanitizeFilePart(token).replace(/^model_/, "") || "main_model",
    phase: null,
    seed: null,
  };
}

function loadModelSpec(input, label, modelPath, phase = null, seed = null) {
  let model = null;
  try {
    const raw = String(readFileSync(modelPath, "utf8") || "").replace(/^\uFEFF/, "");
    model = JSON.parse(raw);
  } catch (err) {
    throw new Error(`failed to parse model JSON (${input}): ${modelPath} (${String(err)})`);
  }
  if (String(model?.format_version || "").trim() !== "neat_python_genome_v1") {
    throw new Error(`invalid model format for ${input}: expected neat_python_genome_v1`);
  }
  return {
    input,
    kind: "model",
    key: label,
    label,
    model,
    modelPath,
    phase,
    seed,
  };
}

function resolveMainModelSpec(token, sideLabel) {
  const artifact = tryResolveMainModelArtifactPath(token);
  if (artifact?.exists) {
    return loadModelSpec(
      token,
      String(artifact.label || sanitizeFilePart(token).replace(/^model_/, "") || "main_model"),
      artifact.path,
      artifact.phase ?? null,
      artifact.seed ?? null
    );
  }
  if (artifact && artifact.exists === false) {
    throw new Error(`model not found for ${token}: ${artifact.path}`);
  }
  throw new Error(
    `invalid ${sideLabel} spec: ${token} (use phase2_seed203, model:path/to/winner_genome.json, hybrid_play(phase2_seed203,H-CL), hybrid_play_go(phase2_seed203,H-CL), or hybrid_play_go(phase2_seed203,H-NEXg,H-CL))`
  );
}

export function resolveHybridPlayModelSpec(rawSpec, sideLabel = "opponent") {
  const token = String(rawSpec || "").trim();
  if (!token) return null;

  const hybridGo = parseHybridPlayGoSpec(token);
  if (hybridGo) {
    const modelSpec = resolveMainModelSpec(hybridGo.modelToken, `${sideLabel}:model`);
    const goStopPolicy = resolveBotPolicy(hybridGo.goStopToken);
    if (!goStopPolicy) {
      throw new Error(
        `invalid ${sideLabel} hybrid go-stop policy: ${hybridGo.goStopToken} (use a policy key from src/ai/policies.js)`
      );
    }
    const heuristicToken = String(hybridGo.heuristicToken || "").trim();
    const heuristicPolicy = heuristicToken ? resolveBotPolicy(heuristicToken) : "";
    if (heuristicToken && !heuristicPolicy) {
      throw new Error(
        `invalid ${sideLabel} hybrid heuristic policy: ${hybridGo.heuristicToken} (use a policy key from src/ai/policies.js)`
      );
    }
    const goStopOnly = !heuristicPolicy;
    return {
      input: token,
      kind: "hybrid_play_model",
      key: goStopOnly
        ? `hybrid_play_go(${modelSpec.key},${goStopPolicy})`
        : `hybrid_play_go(${modelSpec.key},${goStopPolicy},${heuristicPolicy})`,
      label: goStopOnly
        ? `hybrid_play_go(${modelSpec.label},${goStopPolicy})`
        : `hybrid_play_go(${modelSpec.label},${goStopPolicy},${heuristicPolicy})`,
      model: modelSpec.model,
      modelPath: modelSpec.modelPath,
      phase: modelSpec.phase,
      seed: modelSpec.seed,
      heuristicPolicy,
      goStopPolicy,
      goStopOnly,
    };
  }

  const hybrid = parseHybridPlaySpec(token);
  if (hybrid) {
    const modelSpec = resolveMainModelSpec(hybrid.modelToken, `${sideLabel}:model`);
    const heuristicPolicy = resolveBotPolicy(hybrid.heuristicToken);
    if (!heuristicPolicy) {
      throw new Error(
        `invalid ${sideLabel} hybrid heuristic policy: ${hybrid.heuristicToken} (use a policy key from src/ai/policies.js)`
      );
    }
    return {
      input: token,
      kind: "hybrid_play_model",
      key: `hybrid_play(${modelSpec.key},${heuristicPolicy})`,
      label: `hybrid_play(${modelSpec.label},${heuristicPolicy})`,
      model: modelSpec.model,
      modelPath: modelSpec.modelPath,
      phase: modelSpec.phase,
      seed: modelSpec.seed,
      heuristicPolicy,
    };
  }

  return null;
}
