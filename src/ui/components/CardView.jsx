import { useEffect, useState } from "react";
import { buildCardUiAssetPath, DEFAULT_CARD_THEME } from "../../cards.js";

/* ============================================================================
 * Card renderer
 * - single card face/back view
 * - theme-aware fallback image handling
 * ========================================================================== */

/* 1) Asset path helpers */
function resolveBackAsset(theme) {
  return buildCardUiAssetPath("back.svg", theme || DEFAULT_CARD_THEME);
}

function resolveFallbackAsset(theme) {
  return buildCardUiAssetPath("fallback.svg", theme || DEFAULT_CARD_THEME);
}

/* 2) Shared card-back element */
export function cardBackView(theme = DEFAULT_CARD_THEME) {
  return (
    <div className="card card-back">
      <img src={resolveBackAsset(theme)} alt="card back" />
    </div>
  );
}

/* 3) Main card component */
export default function CardView({
  card,
  interactive = false,
  onClick = null,
  forceBack = false,
  t = null,
  theme = DEFAULT_CARD_THEME
}) {
  const [src, setSrc] = useState(forceBack ? resolveBackAsset(theme) : card?.asset || resolveFallbackAsset(theme));
  const month = card?.month ?? "?";
  const name = card?.name ?? (t ? t("card.alt.defaultName") : "Card");
  const altText = t ? t("card.alt.format", { month, name }) : `${month} ${name}`;

  // Sync source when card/theme/back state changes.
  useEffect(() => {
    setSrc(forceBack ? resolveBackAsset(theme) : card?.asset || resolveFallbackAsset(theme));
  }, [card?.asset, forceBack, theme]);

  return (
    <div className={`card${interactive ? " card-selectable" : ""}`} onClick={interactive ? onClick : undefined}>
      <img
        src={src}
        alt={altText}
        onError={() => {
          if (forceBack) {
            setSrc(resolveBackAsset(DEFAULT_CARD_THEME));
            return;
          }
          setSrc(resolveFallbackAsset(DEFAULT_CARD_THEME));
        }}
      />
    </div>
  );
}
