export function resolveMatch({ card, board, source, isLastHandTurn = false, playedMonth = null }) {
  const matches = board.filter((c) => c.month === card.month);
  const result = {
    type: "NONE",
    source,
    eventTag: "NORMAL",
    matches,
    needsChoice: false
  };

  if (matches.length === 0) {
    if (
      source === "flip" &&
      !isLastHandTurn &&
      playedMonth != null &&
      playedMonth === card.month
    ) {
      result.eventTag = "JJOB";
    }
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
      result.eventTag = "DDADAK";
    }
    return result;
  }

  if (matches.length === 2) {
    result.type = "TWO";
    result.needsChoice = matches[0].category !== matches[1].category;
    if (source === "flip" && !isLastHandTurn) result.eventTag = "PPUK";
    return result;
  }

  result.type = "THREE_PLUS";
  if (source === "hand") result.eventTag = "TTAK";
  if (source === "flip" && !isLastHandTurn) result.eventTag = "PPUK";
  return result;
}
