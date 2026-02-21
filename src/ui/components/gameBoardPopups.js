import readmeRaw from "../../../README.md?raw";
import { DEFAULT_LANGUAGE, makeTranslator } from "../i18n/i18n.js";

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatInlineMarkdown(text) {
  const escaped = escapeHtml(text);
  return escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
}

function popupBlockedAlert(t) {
  window.alert(t("popup.blocked"));
}

function renderCards(cards) {
  return cards
    .map((card) => {
      const id = card.cardID || card.id || "";
      const meta = [
        `month:${card.month}`,
        `category:${card.category}`,
        Number.isFinite(Number(card.piValue)) ? `piValue:${card.piValue}` : ""
      ]
        .filter(Boolean)
        .join(" / ");

      return `
        <div class="item">
          <img src="${escapeHtml(card.asset)}" alt="${escapeHtml(card.name)}" />
          <div class="label">${formatInlineMarkdown(`${id} ${card.name}`)}</div>
          <div class="detail">${formatInlineMarkdown(meta)}</div>
        </div>
      `;
    })
    .join("");
}

function buildRulesHtmlFromReadme(markdown, fromSection = 1, toSection = 9) {
  const lines = String(markdown || "").replace(/^\uFEFF/, "").split(/\r?\n/);
  const html = [];
  let inRange = false;
  let listType = null;

  const closeList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = null;
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    const h2 = line.match(/^##\s+(\d+)\.\s*(.+)$/);
    if (h2) {
      const sectionNo = Number(h2[1]);
      if (sectionNo >= fromSection && sectionNo <= toSection) {
        inRange = true;
        closeList();
        html.push(`<h2>${formatInlineMarkdown(`${h2[1]}. ${h2[2]}`)}</h2>`);
        continue;
      }
      if (inRange && sectionNo > toSection) break;
      inRange = false;
      continue;
    }

    if (!inRange) continue;

    const h3 = line.match(/^###\s+(.+)$/);
    if (h3) {
      closeList();
      html.push(`<h3>${formatInlineMarkdown(h3[1])}</h3>`);
      continue;
    }

    const ul = line.match(/^-+\s+(.+)$/);
    if (ul) {
      if (listType !== "ul") {
        closeList();
        listType = "ul";
        html.push("<ul>");
      }
      html.push(`<li>${formatInlineMarkdown(ul[1])}</li>`);
      continue;
    }

    const ol = line.match(/^\d+\.\s+(.+)$/);
    if (ol) {
      if (listType !== "ol") {
        closeList();
        listType = "ol";
        html.push("<ol>");
      }
      html.push(`<li>${formatInlineMarkdown(ol[1])}</li>`);
      continue;
    }

    if (!line.trim()) {
      closeList();
      continue;
    }

    closeList();
    html.push(`<p>${formatInlineMarkdown(line.trim())}</p>`);
  }

  closeList();
  return html.join("");
}

const README_RULES_HTML = buildRulesHtmlFromReadme(readmeRaw, 1, 9);

function getPopupTranslator(language) {
  return makeTranslator(language || DEFAULT_LANGUAGE);
}

export function openCardGuidePopup({ monthCards, bonusCards, language = DEFAULT_LANGUAGE }) {
  const t = getPopupTranslator(language);
  const popup = window.open("", "kflower-card-guide", "width=1080,height=780,resizable=yes,scrollbars=yes");
  if (!popup) {
    popupBlockedAlert(t);
    return;
  }

  popup.document.write(`
    <!doctype html>
    <html lang="${escapeHtml(language)}">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>${escapeHtml(t("popup.cardGuide.title"))}</title>
        <style>
          body { margin: 0; padding: 16px; font-family: "Noto Sans KR", sans-serif; background: #143d16; color: #ecffd6; }
          h1 { margin: 0 0 10px; font-size: 22px; }
          h2 { margin: 14px 0 8px; font-size: 16px; }
          .meta { font-size: 12px; color: #d5f6bf; margin-bottom: 8px; }
          .grid { display: grid; gap: 8px; }
          .grid-main { grid-template-columns: repeat(8, minmax(76px, 1fr)); }
          .grid-small { grid-template-columns: repeat(2, minmax(76px, 1fr)); max-width: 220px; }
          .item { display: grid; justify-items: center; gap: 3px; background: rgba(12, 62, 19, 0.45); border: 1px solid rgba(220, 255, 183, 0.25); border-radius: 8px; padding: 6px; }
          .item img { width: 54px; height: 78px; object-fit: contain; }
          .label { font-size: 11px; color: #e8ffd2; text-align: center; font-weight: 600; line-height: 1.2; }
          .detail { font-size: 10px; color: #c9ebb8; text-align: center; line-height: 1.2; min-height: 1.2em; }
          @media (max-width: 980px) {
            .grid-main { grid-template-columns: repeat(4, minmax(76px, 1fr)); }
          }
        </style>
      </head>
      <body>
        <h1>${escapeHtml(t("popup.cardGuide.title"))}</h1>
        <div class="meta">${escapeHtml(t("popup.cardGuide.meta"))}</div>
        <h2>${escapeHtml(t("popup.cardGuide.mainCards"))}</h2>
        <div class="grid grid-main">${renderCards(monthCards)}</div>
        <h2>${escapeHtml(t("popup.cardGuide.bonusCards"))}</h2>
        <div class="grid grid-small">${renderCards(bonusCards)}</div>
      </body>
    </html>
  `);
  popup.document.close();
  popup.focus();
}

export function openRulesPopup({ language = DEFAULT_LANGUAGE } = {}) {
  const t = getPopupTranslator(language);
  const popup = window.open("", "kflower-rules", "width=980,height=760,resizable=yes,scrollbars=yes");
  if (!popup) {
    popupBlockedAlert(t);
    return;
  }

  const rulesHtml =
    README_RULES_HTML || `<p>${escapeHtml(t("popup.rules.readmeFallback"))}</p>`;

  popup.document.write(`
    <!doctype html>
    <html lang="${escapeHtml(language)}">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>${escapeHtml(t("popup.rules.title"))}</title>
        <style>
          body { margin: 0; padding: 16px; font-family: "Noto Sans KR", sans-serif; background: #143d16; color: #ecffd6; line-height: 1.5; }
          h1 { margin: 0 0 8px; font-size: 22px; }
          h2 { margin: 14px 0 6px; font-size: 16px; color: #f4ffd6; }
          h3 { margin: 10px 0 4px; font-size: 14px; color: #ddffc9; }
          p, li { font-size: 13px; }
          ul, ol { margin: 4px 0 8px; padding-left: 18px; }
          code { background: rgba(8, 27, 12, 0.7); color: #d8ffc7; border: 1px solid rgba(220, 255, 183, 0.25); border-radius: 6px; padding: 1px 5px; }
          .meta { font-size: 12px; color: #d5f6bf; margin-bottom: 12px; }
          .rule-wrap { display: grid; gap: 2px; }
        </style>
      </head>
      <body>
        <h1>${escapeHtml(t("popup.rules.title"))}</h1>
        <div class="meta">${escapeHtml(t("popup.rules.meta"))}</div>
        <div class="rule-wrap">${rulesHtml}</div>
      </body>
    </html>
  `);
  popup.document.close();
  popup.focus();
}

function stripLogPattern(text) {
  return String(text || "")
    .replace(/\{[^}]+\}/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function hiddenPatternsForLogs(language = DEFAULT_LANGUAGE) {
  const translators = [
    getPopupTranslator(language),
    getPopupTranslator("ko"),
    getPopupTranslator("en")
  ];
  const patternKeys = [
    ["log.gameStartRule", { ruleName: "" }],
    ["log.playOneMatchCapture", { player: "", month: "" }],
    ["log.flipMatchCapture", { month: "" }],
    ["log.playTwoMatchWait", { player: "", month: "" }],
    ["log.flipTwoMatchWait", { month: "" }],
    ["pending.match.player", { player: "" }],
    ["pending.match.flip", { player: "" }],
    ["popup.gameLog.hidden.starterCandidate", {}],
    ["popup.gameLog.hidden.matchCapture", {}],
    ["popup.gameLog.hidden.cardSelect", {}]
  ];

  const set = new Set();
  translators.forEach((tr) => {
    patternKeys.forEach(([key, params]) => {
      const pattern = stripLogPattern(tr(key, params));
      if (pattern) set.add(pattern);
    });
  });
  return Array.from(set);
}

export function openGameLogPopup({ log = [], language = DEFAULT_LANGUAGE }) {
  const t = getPopupTranslator(language);
  const popup = window.open("", "kflower-game-log", "width=980,height=760,resizable=yes,scrollbars=yes");
  if (!popup) {
    popupBlockedAlert(t);
    return;
  }

  const visibleLogs = log.filter(
    (line) => !hiddenPatternsForLogs(language).some((pattern) => String(line).includes(pattern))
  );
  const logsHtml =
    visibleLogs.length > 0
      ? visibleLogs.map((line) => `<div class="log-line">${escapeHtml(line)}</div>`).join("")
      : `<div class="empty">${escapeHtml(t("popup.gameLog.empty"))}</div>`;

  popup.document.write(`
    <!doctype html>
    <html lang="${escapeHtml(language)}">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>${escapeHtml(t("popup.gameLog.title"))}</title>
        <style>
          body { margin: 0; padding: 16px; font-family: "Noto Sans KR", sans-serif; background: #0f2f12; color: #eaffd5; }
          h1 { margin: 0 0 10px; font-size: 22px; }
          .meta { font-size: 12px; color: #bee4b3; margin-bottom: 10px; }
          .log-wrap { border: 1px solid rgba(141, 218, 255, 0.55); border-radius: 8px; background: rgba(5, 22, 7, 0.72); padding: 8px; }
          .log-line { padding: 4px 6px; border-bottom: 1px solid rgba(130, 180, 138, 0.26); font-size: 13px; line-height: 1.35; white-space: pre-wrap; word-break: break-word; }
          .log-line:last-child { border-bottom: 0; }
          .empty { padding: 8px; color: #b8c9b4; font-size: 13px; }
        </style>
      </head>
      <body>
        <h1>${escapeHtml(t("popup.gameLog.title"))}</h1>
        <div class="meta">${escapeHtml(t("popup.gameLog.meta", { count: visibleLogs.length }))}</div>
        <div class="log-wrap">${logsHtml}</div>
      </body>
    </html>
  `);
  popup.document.close();
  popup.focus();
}
