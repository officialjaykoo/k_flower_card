import { useEffect, useState } from "react";

export function cardBackView() {
  return (
    <div className="card card-back">
      <img src="/cards/back.svg" alt="card back" />
    </div>
  );
}

export default function CardView({ card, interactive = false, onClick = null, forceBack = false }) {
  const [src, setSrc] = useState(forceBack ? "/cards/back.svg" : (card?.asset || "/cards/fallback.svg"));

  useEffect(() => {
    setSrc(forceBack ? "/cards/back.svg" : (card?.asset || "/cards/fallback.svg"));
  }, [card?.asset, forceBack]);

  return (
    <div className={`card${interactive ? " card-selectable" : ""}`} onClick={interactive ? onClick : undefined}>
      <img
        src={src}
        alt={`${card?.month ?? "?"}월 ${card?.name ?? "카드"}`}
        onError={() => setSrc("/cards/fallback.svg")}
      />
    </div>
  );
}
