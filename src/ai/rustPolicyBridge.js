let NODE_BRIDGE_RUNTIME_PROMISE = null;

const BRIDGE_CACHE = new WeakMap();
const LIVE_CLIENTS = new Set();
const BRIDGE_RUNTIME_STATS = {
  bridge_calls: 0,
  bridge_ok: 0,
  by_backend: {},
  last_backend_used: null,
  last_error: null,
};

function isNodeRuntime() {
  return typeof process !== "undefined" && !!process?.versions?.node;
}

function getEnvVar(name) {
  if (!isNodeRuntime()) {
    return "";
  }
  return String(process.env?.[name] || "").trim();
}

async function loadNodeBridgeRuntime() {
  if (!isNodeRuntime()) {
    return null;
  }
  if (!NODE_BRIDGE_RUNTIME_PROMISE) {
    NODE_BRIDGE_RUNTIME_PROMISE = Promise.all([
      import("node:fs"),
      import("node:path"),
      import("node:readline"),
      import("node:url"),
      import("node:child_process"),
    ]).then(([fs, path, readline, url, childProcess]) => {
      const repoRoot = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..", "..");
      return {
        fs,
        path,
        readline,
        url,
        spawn: childProcess.spawn,
        spawnSync: childProcess.spawnSync,
        repoRoot,
        neatRustRoot: path.join(repoRoot, "neat-rust"),
      };
    });
  }
  return NODE_BRIDGE_RUNTIME_PROMISE;
}

function debugNativeBridge(message) {
  if (!isNodeRuntime()) {
    return;
  }
  const filePath = getEnvVar("NEAT_DEBUG_NATIVE_FILE");
  if (filePath) {
    loadNodeBridgeRuntime()
      .then((runtime) => {
        try {
          runtime?.fs?.appendFileSync?.(filePath, `[rustPolicyBridge] ${message}\n`, "utf8");
        } catch {}
      })
      .catch(() => {});
    return;
  }
  if (getEnvVar("NEAT_DEBUG_NATIVE") === "1") {
    console.error(`[rustPolicyBridge] ${message}`);
  }
}

function normalizeBackend(value) {
  const raw = String(value || "off").trim().toLowerCase();
  if (!raw || raw === "off" || raw === "disabled" || raw === "false") return "off";
  if (raw === "auto") return "auto";
  if (raw === "cpu" || raw === "rust_cpu") return "cpu";
  if (raw === "cuda" || raw === "cuda_native" || raw === "native_cuda" || raw === "rust_cuda") {
    return "cuda_native";
  }
  throw new Error(`invalid native inference backend: ${value}`);
}

function sanitizeFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0.0;
}

function sanitizeInputRows(inputs) {
  if (!Array.isArray(inputs)) return [];
  return inputs.map((row) => {
    if (!Array.isArray(row)) return [];
    return row.map((value) => sanitizeFiniteNumber(value));
  });
}

function sanitizeSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== "object") {
    return null;
  }
  const sourceNodeValues =
    snapshot.nodeValues && typeof snapshot.nodeValues === "object" ? snapshot.nodeValues : {};
  const nodeValues = {};
  for (const [nodeId, value] of Object.entries(sourceNodeValues)) {
    nodeValues[String(nodeId)] = sanitizeFiniteNumber(value);
  }
  return { nodeValues };
}

function noteBridgeSuccess(stats, backendUsed) {
  if (!stats || typeof stats !== "object") {
    return;
  }
  const backendKey = String(backendUsed || "unknown").trim() || "unknown";
  stats.bridge_ok = Number(stats.bridge_ok || 0) + 1;
  stats.request_count = Number(stats.request_count || 0) + 1;
  if (!stats.by_backend || typeof stats.by_backend !== "object") {
    stats.by_backend = {};
  }
  stats.by_backend[backendKey] = Number(stats.by_backend[backendKey] || 0) + 1;
  stats.last_backend_used = backendKey;
}

function noteGlobalBridgeCall() {
  BRIDGE_RUNTIME_STATS.bridge_calls = Number(BRIDGE_RUNTIME_STATS.bridge_calls || 0) + 1;
}

function noteGlobalBridgeSuccess(backendUsed) {
  const backendKey = String(backendUsed || "unknown").trim() || "unknown";
  BRIDGE_RUNTIME_STATS.bridge_ok = Number(BRIDGE_RUNTIME_STATS.bridge_ok || 0) + 1;
  if (!BRIDGE_RUNTIME_STATS.by_backend || typeof BRIDGE_RUNTIME_STATS.by_backend !== "object") {
    BRIDGE_RUNTIME_STATS.by_backend = {};
  }
  BRIDGE_RUNTIME_STATS.by_backend[backendKey] =
    Number(BRIDGE_RUNTIME_STATS.by_backend[backendKey] || 0) + 1;
  BRIDGE_RUNTIME_STATS.last_backend_used = backendKey;
  BRIDGE_RUNTIME_STATS.last_error = null;
}

function noteGlobalBridgeError(err) {
  BRIDGE_RUNTIME_STATS.last_error = String(err && err.message ? err.message : err);
}

export function getRustPolicyBridgeStats() {
  return {
    bridge_calls: Number(BRIDGE_RUNTIME_STATS.bridge_calls || 0),
    bridge_ok: Number(BRIDGE_RUNTIME_STATS.bridge_ok || 0),
    by_backend: { ...(BRIDGE_RUNTIME_STATS.by_backend || {}) },
    last_backend_used: BRIDGE_RUNTIME_STATS.last_backend_used || null,
    last_error: BRIDGE_RUNTIME_STATS.last_error || null,
  };
}

export function resetRustPolicyBridgeStats() {
  BRIDGE_RUNTIME_STATS.bridge_calls = 0;
  BRIDGE_RUNTIME_STATS.bridge_ok = 0;
  BRIDGE_RUNTIME_STATS.by_backend = {};
  BRIDGE_RUNTIME_STATS.last_backend_used = null;
  BRIDGE_RUNTIME_STATS.last_error = null;
}

function candidateBridgePaths(runtime) {
  const root = runtime.neatRustRoot;
  return [
    getEnvVar("NEAT_RUST_POLICY_BRIDGE_EXE"),
    runtime.path.join(root, "target", "x86_64-pc-windows-gnullvm", "debug", "neat-policy-bridge-rs.exe"),
    runtime.path.join(root, "target", "debug", "neat-policy-bridge-rs.exe"),
    runtime.path.join(root, "target", "x86_64-pc-windows-gnullvm", "release", "neat-policy-bridge-rs.exe"),
    runtime.path.join(root, "target", "release", "neat-policy-bridge-rs.exe"),
  ].filter(Boolean);
}

function resolveBridgeExecutable(runtime) {
  const found = candidateBridgePaths(runtime).find((candidate) => runtime.fs.existsSync(candidate));
  if (found) return found;

  const runCargo = runtime.path.join(runtime.neatRustRoot, "run-cargo.ps1");
  const built = runtime.spawnSync(
    "powershell",
    [
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      runCargo,
      "build",
      "--bin",
      "neat-policy-bridge-rs",
    ],
    {
      cwd: runtime.neatRustRoot,
      encoding: "utf8",
    }
  );
  if (built.status !== 0) {
    throw new Error(
      `failed to build neat-policy-bridge-rs: ${String(built.stderr || built.stdout || "").trim()}`
    );
  }
  const afterBuild = candidateBridgePaths(runtime).find((candidate) => runtime.fs.existsSync(candidate));
  if (!afterBuild) {
    throw new Error("neat-policy-bridge-rs executable was not found after build");
  }
  return afterBuild;
}

function serializeCompiledModel(compiled) {
  if (!compiled || String(compiled?.kind || "").trim() !== "neat_python_genome_v1") {
    return null;
  }
  const networkType = String(compiled?.networkType || "feedforward").trim().toLowerCase();
  const orderedNodeIds =
    networkType === "recurrent"
      ? (Array.isArray(compiled?.recurrentNodeIds) ? compiled.recurrentNodeIds : [])
      : (Array.isArray(compiled?.order) ? compiled.order : []);
  const nodeIds = orderedNodeIds.map((nodeId) => Number(nodeId));
  const nodeIndex = new Map(nodeIds.map((nodeId, index) => [nodeId, index]));
  const outputIndices = (Array.isArray(compiled?.outputKeys) ? compiled.outputKeys : []).map((key) => {
    const index = nodeIndex.get(Number(key));
    if (index === undefined) {
      throw new Error(`compiled output key ${key} is missing from ordered node ids`);
    }
    return index;
  });

  const nodeEvals = nodeIds.map((nodeId) => {
    const node = compiled?.nodes?.get(nodeId) || {
      activation: "tanh",
      aggregation: "sum",
      bias: 0,
      response: 1,
      memoryGateEnabled: false,
      memoryGateBias: 0,
      memoryGateResponse: 1,
    };
    const incomingRaw = compiled?.incoming?.get(nodeId) || [];
    const incoming = incomingRaw.map((conn) => {
      const inNode = Number(conn?.in_node || 0);
      if (compiled?.inputSet?.has?.(inNode)) {
        const sourceIndex = (compiled?.inputKeys || []).findIndex((key) => Number(key) === inNode);
        if (sourceIndex < 0) {
          throw new Error(`compiled input key ${inNode} is missing from inputKeys`);
        }
        return {
          sourceKind: "input",
          sourceIndex,
          weight: Number(conn?.weight || 0),
        };
      }
      const sourceIndex = nodeIndex.get(inNode);
      if (sourceIndex === undefined) {
        throw new Error(`compiled node key ${inNode} is missing from ordered node ids`);
      }
      return {
        sourceKind: "node",
        sourceIndex,
        weight: Number(conn?.weight || 0),
      };
    });
    return {
      nodeId,
      activation: String(node?.activation || "tanh"),
      aggregation: String(node?.aggregation || "sum"),
      bias: Number(node?.bias || 0),
      response: Number(node?.response ?? 1),
      memoryGateEnabled: !!node?.memoryGateEnabled,
      memoryGateBias: Number(node?.memoryGateBias || 0),
      memoryGateResponse: Number(node?.memoryGateResponse ?? 1),
      incoming,
    };
  });

  return {
    networkType,
    inputCount: Array.isArray(compiled?.inputKeys) ? compiled.inputKeys.length : 0,
    outputIndices,
    nodeEvals,
  };
}

class RustPolicyBridgeClient {
  constructor(runtime, executablePath, backend, compiledSpec) {
    this.runtime = runtime;
    this.child = runtime.spawn(executablePath, ["--backend", backend], {
      cwd: runtime.neatRustRoot,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.pending = new Map();
    this.nextId = 1;
    this.exited = false;
    this.stderr = "";
    LIVE_CLIENTS.add(this);

    const rl = runtime.readline.createInterface({ input: this.child.stdout });
    rl.on("line", (line) => {
      if (!line) return;
      let parsed = null;
      try {
        parsed = JSON.parse(line);
      } catch (err) {
        this.failAll(`invalid bridge response JSON: ${String(err?.message || err)}`);
        return;
      }
      const id = Number(parsed?.id || 0);
      const pending = this.pending.get(id);
      if (!pending) return;
      this.pending.delete(id);
      if (parsed?.ok) {
        pending.resolve(parsed);
      } else {
        pending.reject(new Error(String(parsed?.error || "unknown bridge error")));
      }
    });
    this.child.stderr.on("data", (chunk) => {
      this.stderr += String(chunk || "");
      if (this.stderr.length > 4000) {
        this.stderr = this.stderr.slice(-4000);
      }
    });
    this.child.stdin.on("error", (err) => {
      debugNativeBridge(`child stdin error err=${String(err && err.message ? err.message : err)}`);
      this.exited = true;
      this.failAll(`bridge stdin error: ${String(err && err.message ? err.message : err)}`);
      LIVE_CLIENTS.delete(this);
    });
    this.child.on("exit", (code, signal) => {
      this.exited = true;
      const suffix = this.stderr ? ` stderr=${this.stderr.trim()}` : "";
      debugNativeBridge(`child exit code=${code} signal=${signal}${suffix}`);
      this.failAll(`bridge exited code=${code} signal=${signal}${suffix}`);
      LIVE_CLIENTS.delete(this);
    });
    this.ready = this.request({ kind: "init", compiled: compiledSpec }).then(() => undefined);
  }

  request(payload) {
    if (this.exited) {
      return Promise.reject(new Error("rust policy bridge is already closed"));
    }
    const id = this.nextId++;
    const message = JSON.stringify({ id, ...payload });
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(`${message}\n`, (err) => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  async evaluate(inputs, snapshot = null) {
    await this.ready;
    return this.request({ kind: "eval", inputs, snapshot });
  }

  async close() {
    if (this.exited) return;
    try {
      await this.request({ kind: "shutdown" });
    } catch {}
    try {
      this.child.kill();
    } catch {}
    this.exited = true;
    LIVE_CLIENTS.delete(this);
  }

  failAll(message) {
    for (const pending of this.pending.values()) {
      pending.reject(new Error(message));
    }
    this.pending.clear();
  }
}

async function getBridgeClient(compiled, backend) {
  const existing = BRIDGE_CACHE.get(compiled);
  if (existing && existing.backend === backend) {
    return existing.client;
  }
  const compiledSpec = serializeCompiledModel(compiled);
  if (!compiledSpec) {
    throw new Error("native policy bridge only supports neat_python_genome_v1 compiled models");
  }
  const runtime = await loadNodeBridgeRuntime();
  if (!runtime) {
    throw new Error("native policy bridge is only available in Node runtimes");
  }
  const executablePath = resolveBridgeExecutable(runtime);
  const client = new RustPolicyBridgeClient(runtime, executablePath, backend, compiledSpec);
  BRIDGE_CACHE.set(compiled, { backend, client });
  return client;
}

export async function evaluateCompiledPolicyBatch(compiled, inputs, snapshot, options = {}) {
  const backend = normalizeBackend(options.nativeInferenceBackend);
  debugNativeBridge(`evaluate start backend=${backend} batch=${Array.isArray(inputs) ? inputs.length : 0}`);
  noteGlobalBridgeCall();
  if (options.nativeInferenceStats && typeof options.nativeInferenceStats === "object") {
    options.nativeInferenceStats.bridge_calls =
      Number(options.nativeInferenceStats.bridge_calls || 0) + 1;
  }
  if (backend === "off" || !isNodeRuntime()) {
    return null;
  }
  try {
    const client = await getBridgeClient(compiled, backend);
    debugNativeBridge("client ready");
    const result = await client.evaluate(sanitizeInputRows(inputs), sanitizeSnapshot(snapshot));
    debugNativeBridge(`client evaluate resolved backendUsed=${String(result?.backendUsed || "")}`);
    noteGlobalBridgeSuccess(result?.backendUsed || backend);
    noteBridgeSuccess(options.nativeInferenceStats, result?.backendUsed || backend);
    return result;
  } catch (err) {
    debugNativeBridge(`client evaluate failed err=${String(err && err.message ? err.message : err)}`);
    noteGlobalBridgeError(err);
    if (options.nativeInferenceStats && typeof options.nativeInferenceStats === "object") {
      options.nativeInferenceStats.last_bridge_error = String(err && err.message ? err.message : err);
    }
    throw err;
  }
}

export async function closeAllRustPolicyBridges() {
  const pending = [];
  for (const client of LIVE_CLIENTS) {
    pending.push(client.close());
  }
  await Promise.allSettled(pending);
}
