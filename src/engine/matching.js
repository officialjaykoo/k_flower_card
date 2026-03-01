/* ============================================================================
 * Match resolution
 * - Determine match type by board month count
 * - Tag events produced by hand/flip context
 * ========================================================================== */

export function resolveMatch({
  card,
  board,
  source,
  isLastHandTurn = false,
  playedMonth = null,
  playedCardId = null
}) {
  const matches = board.filter((c) => c.month === card.month);
  const result = {
    type: "NONE",
    source,
    eventTag: "NORMAL",
    matches,
    needsChoice: false
  };

  if (matches.length === 0) {
    return result;
  }

  if (matches.length === 1) {
    result.type = "ONE";
    if (
      source === "flip" &&
      !isLastHandTurn &&
      playedMonth != null &&
      playedMonth === card.month
    ) {
      const matchedPlayedCard = playedCardId != null && matches[0]?.id === playedCardId;
      result.eventTag = matchedPlayedCard ? "JJOB" : "DDADAK";
    }
    return result;
  }

  if (matches.length === 2) {
    result.type = "TWO";
    result.needsChoice = matches[0].category !== matches[1].category;
    return result;
  }

  result.type = "THREE_PLUS";
  if (source === "hand" && !isLastHandTurn) result.eventTag = "PANSSEUL";
  return result;
}
