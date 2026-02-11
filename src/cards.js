const cardCatalog = [
  { month: 1, category: "kwang", name: "Pine Bright" },
  { month: 1, category: "ribbon", name: "Poetry Ribbon" },
  { month: 1, category: "five", name: "Crane" },
  { month: 1, category: "junk", name: "Pine Junk" },

  { month: 2, category: "ribbon", name: "Poetry Ribbon" },
  { month: 2, category: "five", name: "Bush Warbler" },
  { month: 2, category: "junk", name: "Plum Junk A" },
  { month: 2, category: "junk", name: "Plum Junk B" },

  { month: 3, category: "kwang", name: "Cherry Bright" },
  { month: 3, category: "ribbon", name: "Poetry Ribbon" },
  { month: 3, category: "junk", name: "Cherry Junk A" },
  { month: 3, category: "junk", name: "Cherry Junk B" },

  { month: 4, category: "ribbon", name: "Plant Ribbon" },
  { month: 4, category: "five", name: "Cuckoo" },
  { month: 4, category: "junk", name: "Wisteria Junk A" },
  { month: 4, category: "junk", name: "Wisteria Junk B" },

  { month: 5, category: "ribbon", name: "Plant Ribbon" },
  { month: 5, category: "five", name: "Bridge" },
  { month: 5, category: "junk", name: "Iris Junk A" },
  { month: 5, category: "junk", name: "Iris Junk B" },

  { month: 6, category: "ribbon", name: "Blue Ribbon" },
  { month: 6, category: "five", name: "Butterflies" },
  { month: 6, category: "junk", name: "Peony Junk" },
  { month: 6, category: "junk", name: "Double Junk", doubleJunk: true },

  { month: 7, category: "five", name: "Boar" },
  { month: 7, category: "ribbon", name: "Plant Ribbon" },
  { month: 7, category: "junk", name: "Clover Junk A" },
  { month: 7, category: "junk", name: "Clover Junk B" },

  { month: 8, category: "kwang", name: "Moon Bright" },
  { month: 8, category: "five", name: "Geese" },
  { month: 8, category: "junk", name: "Pampas Junk A" },
  { month: 8, category: "junk", name: "Pampas Junk B" },

  { month: 9, category: "five", name: "Sake Cup" },
  { month: 9, category: "ribbon", name: "Blue Ribbon" },
  { month: 9, category: "junk", name: "Chrysanthemum Junk A" },
  { month: 9, category: "junk", name: "Chrysanthemum Junk B" },

  { month: 10, category: "kwang", name: "Rain Bright" },
  { month: 10, category: "five", name: "Deer" },
  { month: 10, category: "junk", name: "Maple Junk A" },
  { month: 10, category: "junk", name: "Maple Junk B" },

  { month: 11, category: "kwang", name: "Willow Bright" },
  { month: 11, category: "five", name: "Swallow" },
  { month: 11, category: "junk", name: "Willow Junk" },
  { month: 11, category: "junk", name: "Double Junk", doubleJunk: true },

  { month: 12, category: "kwang", name: "Paulownia Bright" },
  { month: 12, category: "five", name: "Rain" },
  { month: 12, category: "junk", name: "Paulownia Junk A" },
  { month: 12, category: "junk", name: "Paulownia Junk B" },

  {
    month: 13,
    category: "junk",
    name: "Bonus Double",
    doubleJunk: true,
    bonus: { stealPi: 1 }
  },
  {
    month: 13,
    category: "junk",
    name: "Bonus Triple",
    tripleJunk: true,
    bonus: { stealPi: 1 }
  }
];

export function buildDeck() {
  return cardCatalog.map((card, idx) => ({
    ...card,
    id: `${card.month}-${idx}`,
    asset: `/cards/${card.month}-${card.name}.png`
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

export function groupByMonth(cards) {
  return cards.reduce((acc, card) => {
    const list = acc[card.month] || [];
    list.push(card);
    acc[card.month] = list;
    return acc;
  }, {});
}
