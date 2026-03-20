const K_HYPERNEAT_EXECUTOR_FORMAT = "k_hyperneat_executor_v1";

function activation(name, x) {
  const n = String(name || "identity").trim().toLowerCase();
  const v = Number(x || 0);
  if (n === "identity" || n === "none") return v;
  if (n === "linear") return Math.min(1.0, Math.max(-1.0, v));
  if (n === "sigmoid") return 1.0 / (1.0 + Math.exp(-v));
  if (n === "relu") return Math.max(0, v);
  if (n === "step") return v > 0 ? 1.0 : 0.0;
  if (n === "softmax") return Math.exp(v);
  if (n === "gaussian") return Math.exp(-((2.5 * v) ** 2));
  if (n === "offsetgaussian") return 2.0 * Math.exp(-((2.5 * v) ** 2)) - 1.0;
  if (n === "sine") return Math.sin(2.0 * v);
  if (n === "cos") return Math.cos(2.0 * v);
  if (n === "square") return v * v;
  if (n === "abs") return Math.abs(v);
  if (n === "exp") return Math.exp(Math.min(1.0, v));
  return Math.tanh(v);
}

export function isKHyperneatExecutorModel(raw) {
  return String(raw?.format_version || "").trim() === K_HYPERNEAT_EXECUTOR_FORMAT;
}

export function compileKHyperneatExecutor(raw) {
  if (!isKHyperneatExecutorModel(raw)) {
    throw new Error(`invalid K-HyperNEAT runtime format: expected ${K_HYPERNEAT_EXECUTOR_FORMAT}`);
  }
  const nodeCount = Math.max(0, Number(raw?.node_count || 0));
  const inputNodeIds = Array.isArray(raw?.input_node_ids) ? raw.input_node_ids.map((v) => Number(v)) : [];
  const outputNodeIds = Array.isArray(raw?.output_node_ids) ? raw.output_node_ids.map((v) => Number(v)) : [];
  const actions = [];
  for (const action of raw?.actions || []) {
    const kind = String(action?.kind || "").trim().toLowerCase();
    if (kind === "link") {
      actions.push({
        kind: "link",
        sourceId: Number(action?.source_id || 0),
        targetId: Number(action?.target_id || 0),
        weight: Number(action?.weight || 0),
      });
      continue;
    }
    if (kind === "activation") {
      actions.push({
        kind: "activation",
        nodeId: Number(action?.node_id || 0),
        bias: Number(action?.bias || 0),
        activation: String(action?.activation || "identity"),
      });
      continue;
    }
    throw new Error(`unknown K-HyperNEAT executor action kind: ${kind}`);
  }
  return {
    formatVersion: K_HYPERNEAT_EXECUTOR_FORMAT,
    nodeCount,
    inputNodeIds,
    outputNodeIds,
    adapter: raw?.adapter && typeof raw.adapter === "object" ? raw.adapter : null,
    actions,
  };
}

export function runKHyperneatExecutor(compiled, inputValues) {
  const nodeCount = Math.max(0, Number(compiled?.nodeCount || 0));
  const values = new Array(nodeCount).fill(0.0);
  const inputs = Array.isArray(inputValues) ? inputValues : [];

  for (let index = 0; index < compiled.inputNodeIds.length; index += 1) {
    const nodeId = Number(compiled.inputNodeIds[index] || 0);
    if (nodeId < 0 || nodeId >= nodeCount) continue;
    values[nodeId] = Number(inputs[index] || 0.0);
  }

  for (const action of compiled.actions || []) {
    if (action.kind === "link") {
      const sourceId = Number(action.sourceId || 0);
      const targetId = Number(action.targetId || 0);
      if (sourceId < 0 || sourceId >= nodeCount || targetId < 0 || targetId >= nodeCount) continue;
      values[targetId] += values[sourceId] * Number(action.weight || 0);
      continue;
    }
    if (action.kind === "activation") {
      const nodeId = Number(action.nodeId || 0);
      if (nodeId < 0 || nodeId >= nodeCount) continue;
      values[nodeId] = activation(action.activation, values[nodeId] + Number(action.bias || 0));
    }
  }

  return compiled.outputNodeIds.map((nodeId) => {
    const value = Number(values[Number(nodeId) || 0] || 0.0);
    return Number.isFinite(value) ? value : 0.0;
  });
}
