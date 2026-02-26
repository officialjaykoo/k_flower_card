/* ============================================================================
 * Rule presets
 * - `A` is the default rule set used across runtime/training scripts.
 * ========================================================================== */
export const ruleSets = {
  A: {
    name: "Unified Project Rules",
    description: "Project default Matgo rules (Go/Stop starts at 7 points)",
    goMinScore: 7,
    bakMultipliers: { gwang: 2, pi: 2, mongBak: 2 },
    useEarlyStop: true
  }
};
