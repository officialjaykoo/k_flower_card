import { useEffect, useState } from "react";
import { buildCardUiAssetPath, DEFAULT_CARD_THEME } from "../../cards.js";

function resolveBackAsset(theme) {
  return buildCardUiAssetPath("back.svg", theme || DEFAULT_CARD_THEME);
}

function resolveFallbackAsset(theme) {
  return buildCardUiAssetPath("fallback.svg", theme || DEFAULT_CARD_THEME);
}

export function cardBackView(theme = DEFAULT_CARD_THEME) {
  return (
    <div className="card card-back">
      <img src={resolveBackAsset(theme)} alt="card back" />
    </div>
  );
}

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
  const name = card?.name ?? (t ? t("card.alt.defaultName") : "카드");
  const altText = t ? t("card.alt.format", { month, name }) : `${month}월 ${name}`;

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
