// Render the deck's slide HTML (source/slides/*.html) to 1920x1080 PNGs.
//   node docs/presentation/tools/render_slides.js <output-dir>
// Uses the Playwright Chromium that is preinstalled in the dev container.
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const out = path.resolve(process.argv[2] || path.join(root, "build"));
const deck = JSON.parse(fs.readFileSync(path.join(root, "source/deck.json"), "utf8"));
const blobs = JSON.parse(fs.readFileSync(path.join(root, "assets/blob-map.json"), "utf8"));

const fonts = Object.values(deck.faces).map((f) => `<link rel="stylesheet" href="${f.href}">`).join("\n");
const css = `
  body { margin: 0; background: #888; }
  section { position: relative; width: 1920px; height: 1080px; box-sizing: border-box; overflow: hidden; margin: 0 0 20px; }
  section > aside { display: none; }
  h1, h2, h3, p, ul, ol, table, hr { margin: 0; }
  h1 { font-size: 96px; font-weight: 600; line-height: 1.1; }
  h2 { font-size: 64px; font-weight: 600; line-height: 1.15; }
  h3 { font-size: 44px; font-weight: 600; line-height: 1.2; }
  p { line-height: 1.4; }
  ul, ol { padding-left: 1.3em; }
  table { border-collapse: collapse; }
  div { box-sizing: border-box; }
  hr { border: 0; width: 100%; }
  x-shape { display: block; flex: none; }
  x-shape[kind="arrow-right"] { clip-path: polygon(0 30%, 60% 30%, 60% 0, 100% 50%, 60% 100%, 60% 70%, 0 70%); }
  x-shape[kind="arrow-left"] { clip-path: polygon(100% 30%, 40% 30%, 40% 0, 0 50%, 40% 100%, 40% 70%, 100% 70%); }
  x-icon { display: block; flex: none; }
`;
const codeIcon = `<svg viewBox="0 0 24 24" width="100%" height="100%" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`;

const slides = deck.order.map((id) => {
  let html = fs.readFileSync(path.join(root, "source/slides", `${id}.html`), "utf8");
  html = html.replace(/\/_blob\/([0-9a-f]{32})/g, (_, b) => "file://" + path.join(root, "assets", blobs[b]));
  html = html.replace(/<x-icon name="Code"([^>]*)><\/x-icon>/g, `<x-icon$1>${codeIcon}</x-icon>`);
  return html;
});

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const page = `<!doctype html><html><head><meta charset="utf-8">${fonts}<style>${css}</style></head><body>${slides.join("\n")}</body></html>`;
  const file = path.join(out, "deck-render.html");
  fs.writeFileSync(file, page);
  const browser = await chromium.launch();
  const tab = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await tab.goto("file://" + file, { waitUntil: "networkidle" });
  await tab.evaluate(() => document.fonts.ready);
  const sections = await tab.$$("section");
  const notes = [];
  for (let i = 0; i < sections.length; i++) {
    const name = path.join(out, `slide-${String(i + 1).padStart(2, "0")}.png`);
    await sections[i].screenshot({ path: name });
    notes.push(await sections[i].evaluate((s) => (s.querySelector(":scope > aside") || {}).textContent || ""));
  }
  fs.writeFileSync(path.join(out, "notes.json"), JSON.stringify(notes, null, 2));
  await browser.close();
  console.log(`Rendered ${sections.length} slides to ${out}`);
})();
