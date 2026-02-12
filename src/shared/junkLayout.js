import { piValue } from "../engine/scoring.js";

export function packJunkRows(cards) {
  const remain = (cards || []).map((card, idx) => ({ card, value: piValue(card), idx }));
  const rows = [];

  const comboPatterns = [
    [3, 2],
    [2, 2, 1],
    [3, 1, 1],
    [2, 1, 1, 1],
    [1, 1, 1, 1, 1]
  ];

  const pickPattern = (pattern) => {
    const picked = [];
    let cursor = -1;
    for (const wanted of pattern) {
      let found = -1;
      for (let i = cursor + 1; i < remain.length; i += 1) {
        if (remain[i].value === wanted) {
          found = i;
          break;
        }
      }
      if (found < 0) return null;
      picked.push(found);
      cursor = found;
    }
    return picked;
  };

  // 5피 줄을 최대한 먼저 만든다. 카드 선택은 획득 순서를 유지한다.
  while (true) {
    let picked = null;
    for (const pattern of comboPatterns) {
      picked = pickPattern(pattern);
      if (picked) break;
    }
    if (!picked) break;

    rows.push(picked.map((idx) => remain[idx].card));
    picked
      .slice()
      .sort((a, b) => b - a)
      .forEach((idx) => {
        remain.splice(idx, 1);
      });
  }

  // 잔여는 2피, 3피 우선으로 모아 최상단에 둔다.
  if (remain.length > 0) {
    const twos = remain.filter((x) => x.value === 2).map((x) => x.card);
    const threes = remain.filter((x) => x.value >= 3).map((x) => x.card);
    const ones = remain.filter((x) => x.value <= 1).map((x) => x.card);
    rows.push([...twos, ...threes, ...ones]);
  }

  return rows;
}

export function junkTopRightOrder(rows) {
  const ids = [];
  for (let r = rows.length - 1; r >= 0; r -= 1) {
    const row = rows[r];
    for (let i = row.length - 1; i >= 0; i -= 1) {
      ids.push(row[i].id);
    }
  }
  return ids;
}

export function junkRowCount(cards) {
  return packJunkRows(cards).length;
}

export function flattenPackedJunk(cards) {
  return packJunkRows(cards).flat();
}
