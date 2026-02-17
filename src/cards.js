const cardCatalog = [
  { cardID: "A0", month: 1, category: "kwang", name: "Pine Bright" },
  { cardID: "A1", month: 1, category: "ribbon", name: "Poetry Ribbon", comboTags: ["redRibbons"] },
  { cardID: "A2", month: 1, category: "junk", name: "Pine Junk A", piValue: 1 },
  { cardID: "A3", month: 1, category: "junk", name: "Pine Junk B", piValue: 1 },

  { cardID: "B0", month: 2, category: "five", name: "Bush Warbler", comboTags: ["fiveBirds"] },
  { cardID: "B1", month: 2, category: "ribbon", name: "Poetry Ribbon", comboTags: ["redRibbons"] },
  { cardID: "B2", month: 2, category: "junk", name: "Plum Junk A", piValue: 1 },
  { cardID: "B3", month: 2, category: "junk", name: "Plum Junk B", piValue: 1 },

  { cardID: "C0", month: 3, category: "kwang", name: "Cherry Bright" },
  { cardID: "C1", month: 3, category: "ribbon", name: "Poetry Ribbon", comboTags: ["redRibbons"] },
  { cardID: "C2", month: 3, category: "junk", name: "Cherry Junk A", piValue: 1 },
  { cardID: "C3", month: 3, category: "junk", name: "Cherry Junk B", piValue: 1 },

  { cardID: "D0", month: 4, category: "five", name: "Cuckoo", comboTags: ["fiveBirds"] },
  { cardID: "D1", month: 4, category: "ribbon", name: "Plant Ribbon", comboTags: ["plainRibbons"] },
  { cardID: "D2", month: 4, category: "junk", name: "Wisteria Junk A", piValue: 1 },
  { cardID: "D3", month: 4, category: "junk", name: "Wisteria Junk B", piValue: 1 },

  { cardID: "E0", month: 5, category: "five", name: "Bridge" },
  { cardID: "E1", month: 5, category: "ribbon", name: "Plant Ribbon", comboTags: ["plainRibbons"] },
  { cardID: "E2", month: 5, category: "junk", name: "Iris Junk A", piValue: 1 },
  { cardID: "E3", month: 5, category: "junk", name: "Iris Junk B", piValue: 1 },

  { cardID: "F0", month: 6, category: "five", name: "Butterflies" },
  { cardID: "F1", month: 6, category: "ribbon", name: "Blue Ribbon", comboTags: ["blueRibbons"] },
  { cardID: "F2", month: 6, category: "junk", name: "Peony Junk A", piValue: 1 },
  { cardID: "F3", month: 6, category: "junk", name: "Peony Junk B", piValue: 1 },

  { cardID: "G0", month: 7, category: "five", name: "Boar" },
  { cardID: "G1", month: 7, category: "ribbon", name: "Plant Ribbon", comboTags: ["plainRibbons"] },
  { cardID: "G2", month: 7, category: "junk", name: "Clover Junk A", piValue: 1 },
  { cardID: "G3", month: 7, category: "junk", name: "Clover Junk B", piValue: 1 },

  { cardID: "H0", month: 8, category: "kwang", name: "Moon Bright" },
  { cardID: "H1", month: 8, category: "five", name: "Geese", comboTags: ["fiveBirds"] },
  { cardID: "H2", month: 8, category: "junk", name: "Pampas Junk A", piValue: 1 },
  { cardID: "H3", month: 8, category: "junk", name: "Pampas Junk B", piValue: 1 },

  { cardID: "I0", month: 9, category: "five", name: "Sake Cup" },
  { cardID: "I1", month: 9, category: "ribbon", name: "Blue Ribbon", comboTags: ["blueRibbons"] },
  { cardID: "I2", month: 9, category: "junk", name: "Chrysanthemum Junk A", piValue: 1 },
  { cardID: "I3", month: 9, category: "junk", name: "Chrysanthemum Junk B", piValue: 1 },

  { cardID: "J0", month: 10, category: "five", name: "Deer" },
  { cardID: "J1", month: 10, category: "ribbon", name: "Maple Ribbon", comboTags: ["blueRibbons"] },
  { cardID: "J2", month: 10, category: "junk", name: "Maple Junk A", piValue: 1 },
  { cardID: "J3", month: 10, category: "junk", name: "Maple Junk B", piValue: 1 },

  { cardID: "K0", month: 11, category: "kwang", name: "Willow Bright" },
  { cardID: "K1", month: 11, category: "junk", name: "Willow Double Junk", piValue: 2 },
  { cardID: "K2", month: 11, category: "junk", name: "Willow Junk A", piValue: 1 },
  { cardID: "K3", month: 11, category: "junk", name: "Willow Junk B", piValue: 1 },

  { cardID: "L0", month: 12, category: "kwang", name: "Paulownia Bright" },
  { cardID: "L1", month: 12, category: "five", name: "Rain Five" },
  { cardID: "L2", month: 12, category: "ribbon", name: "Paulownia Ribbon" },
  { cardID: "L3", month: 12, category: "junk", name: "Paulownia Junk", piValue: 2 },

  {
    cardID: "M0",
    month: 13,
    category: "junk",
    name: "Bonus Double",
    piValue: 2,
    bonus: { stealPi: 1 }
  },
  {
    cardID: "M1",
    month: 13,
    category: "junk",
    name: "Bonus Triple",
    piValue: 3,
    bonus: { stealPi: 1 }
  }
];

export const DEFAULT_CARD_THEME = "original";
export const CARD_THEMES = Object.freeze(["original", "k-flower"]);

export function normalizeCardTheme(theme) {
  return CARD_THEMES.includes(theme) ? theme : DEFAULT_CARD_THEME;
}

export function buildCardAssetPath(cardId, theme = DEFAULT_CARD_THEME) {
  return `/cards/${normalizeCardTheme(theme)}/${cardId}.svg`;
}

export function buildCardUiAssetPath(filename, theme = DEFAULT_CARD_THEME) {
  return `/cards/${normalizeCardTheme(theme)}/${filename}`;
}

export function buildDeck(theme = DEFAULT_CARD_THEME) {
  const cardTheme = normalizeCardTheme(theme);
  return cardCatalog.map((card) => ({
    ...card,
    id: card.cardID,
    asset: buildCardAssetPath(card.cardID, cardTheme)
  }));
}

export function shuffle(deck, rng = Math.random) {
  const arr = deck.slice();
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
