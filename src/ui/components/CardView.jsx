import { useEffect, useState } from "react";

export function cardBackView() {
  return (
    <div className="card card-back">
      <img src="/cards/back.svg" alt="card back" />
    </div>
  );
}

export default function CardView({ card, interactive = false, onClick = null, forceBack = false, t = null }) {
  const [src, setSrc] = useState(forceBack ? "/cards/back.svg" : card?.asset || "/cards/fallback.svg");
  const month = card?.month ?? "?";
  const name = card?.name ?? (t ? t("card.alt.defaultName") : "카드");
  const altText = t ? t("card.alt.format", { month, name }) : `${month}월 ${name}`;

  useEffect(() => {
    setSrc(forceBack ? "/cards/back.svg" : card?.asset || "/cards/fallback.svg");
  }, [card?.asset, forceBack]);

  return (
    <div className={`card${interactive ? " card-selectable" : ""}`} onClick={interactive ? onClick : undefined}>
      <img
        src={src}
        alt={altText}
        onError={() => setSrc("/cards/fallback.svg")}
      />
    </div>
  );
}
