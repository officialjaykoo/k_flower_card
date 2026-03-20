import readline from "node:readline";
import { runModelDuelCli } from "./model_duel_worker.mjs";

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

for await (const rawLine of rl) {
  const line = String(rawLine || "").trim();
  if (!line) continue;

  let request = null;
  let id = null;
  try {
    request = JSON.parse(line);
    id = request?.id ?? null;
    if (!Array.isArray(request?.argv)) {
      throw new Error("request.argv must be an array");
    }
    const summary = runModelDuelCli(request.argv, {
      writeStdout: false,
      writeResultFile: false,
    });
    process.stdout.write(`${JSON.stringify({ id, ok: true, summary })}\n`);
  } catch (err) {
    const payload = {
      id,
      ok: false,
      error: {
        message: String(err?.message || err),
        stack: String(err?.stack || ""),
      },
    };
    process.stdout.write(`${JSON.stringify(payload)}\n`);
  }
}
