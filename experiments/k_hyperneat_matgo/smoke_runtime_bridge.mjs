import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createSeededRng, initSimulationGame } from "../../src/engine/index.js";
import { compileKHyperneatExecutor, runKHyperneatExecutor } from "../../src/ai/kHyperneatExecutor.js";
import { encodeMatgoStateToKHyperneatInputs } from "../../src/ai/kHyperneatMatgoAdapter.js";

const runtimePath = resolve("experiments/k_hyperneat_matgo/smoke_runtime.json");
const raw = JSON.parse(String(readFileSync(runtimePath, "utf8") || "").replace(/^\uFEFF/, ""));
const compiled = compileKHyperneatExecutor(raw);
const state = initSimulationGame("A", createSeededRng("k-hyperneat-smoke"), {
  firstTurnKey: "human",
  kiboDetail: "none",
});
const inputs = encodeMatgoStateToKHyperneatInputs(state, "human", compiled);
const outputs = runKHyperneatExecutor(compiled, inputs);

console.log(JSON.stringify({
  nodeCount: compiled.nodeCount,
  actionCount: compiled.actions.length,
  inputCount: inputs.length,
  outputCount: outputs.length,
  adapterKind: compiled.adapter?.kind || null,
  outputs,
}, null, 2));
