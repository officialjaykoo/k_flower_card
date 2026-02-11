import { useState } from "react";

export function cardBackView() {
  return <div className="card card-back">HWA</div>;
}

export default function CardView({ card, interactive = false, onClick = null }) {
  const [broken, setBroken] = useState(false);
  return (
    <div className={`card${interactive ? " card-selectable" : ""}`} onClick={interactive ? onClick : undefined}>
      {broken || !card.asset ? (
        <div>{`${card.month}월 ${card.name}`}</div>
      ) : (
        <img src={card.asset} alt={`${card.month}월 ${card.name}`} onError={() => setBroken(true)} />
      )}
    </div>
  );
}

