import fs from "fs/promises";
import path from "path";
import sharp from "sharp";

const cardsDir = path.resolve("public/cards");
const MONTHS = Array.from({ length: 12 }, (_, i) => i + 1);
const MONTH_ID_PREFIX = {
  1: "A",
  2: "B",
  3: "C",
  4: "D",
  5: "E",
  6: "F",
  7: "G",
  8: "H",
  9: "I",
  10: "J",
  11: "K",
  12: "L"
};

function isRed(r, g, b, a) {
  return a > 120 && r > 140 && g < 120 && b < 120 && r - g > 40 && r - b > 40;
}

async function detectCardBoxes(srcPath) {
  const { data, info } = await sharp(srcPath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height, channels } = info;
  const visited = new Uint8Array(width * height);
  const comps = [];

  const index = (x, y) => y * width + x;
  const isRedAt = (x, y) => {
    const i = (y * width + x) * channels;
    return isRed(data[i], data[i + 1], data[i + 2], data[i + 3]);
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const k = index(x, y);
      if (visited[k] || !isRedAt(x, y)) continue;

      visited[k] = 1;
      const q = [[x, y]];
      let qi = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;

      while (qi < q.length) {
        const [cx, cy] = q[qi];
        qi += 1;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1]
        ];

        for (const [nx, ny] of neighbors) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const ni = index(nx, ny);
          if (visited[ni] || !isRedAt(nx, ny)) continue;
          visited[ni] = 1;
          q.push([nx, ny]);
        }
      }

      const w = maxX - minX + 1;
      const h = maxY - minY + 1;
      comps.push({ minX, minY, w, h });
    }
  }

  const boxes = comps
    .filter((c) => c.h >= height * 0.75 && c.w >= width * 0.12)
    .sort((a, b) => a.minX - b.minX)
    .slice(0, 4);

  if (boxes.length !== 4) {
    throw new Error(`Expected 4 card boxes, got ${boxes.length}`);
  }

  return boxes;
}

async function writeSvgFromCrop(srcPath, box, outPath) {
  const png = await sharp(srcPath).extract({ left: box.minX, top: box.minY, width: box.w, height: box.h }).png().toBuffer();
  const b64 = png.toString("base64");
  const svg =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<svg xmlns="http://www.w3.org/2000/svg" width="${box.w}" height="${box.h}" viewBox="0 0 ${box.w} ${box.h}">\n` +
    `  <image href="data:image/png;base64,${b64}" width="${box.w}" height="${box.h}"/>\n` +
    `</svg>\n`;
  await fs.writeFile(outPath, svg, "utf8");
}

for (const month of MONTHS) {
  const srcPath = path.join(cardsDir, `${month}.webp`);
  const boxes = await detectCardBoxes(srcPath);
  const prefix = MONTH_ID_PREFIX[month];

  if (!prefix) {
    throw new Error(`Missing card ID prefix for month ${month}`);
  }

  for (let i = 0; i < 4; i += 1) {
    const outPath = path.join(cardsDir, `${prefix}${i}.svg`);
    await writeSvgFromCrop(srcPath, boxes[i], outPath);
  }

  const summary = boxes.map((b) => `[x=${b.minX},y=${b.minY},w=${b.w},h=${b.h}]`).join(" ");
  console.log(`${month}.webp -> ${prefix}0..${prefix}3 ${summary}`);
}

