/* Interactive VBI explorer (Phase 3): a truck crosses a realistic 40 m damped
 * overpass; slide the speed and watch the deflection, DAF and B-WIM readout.
 * Vanilla JS + inline SVG; data injected via output/explore_data.js. */

const D = EXPLORE_DATA;
const M = D.meta;
const NX = D.node_x;
const SCN = D.scenarios;

let cur = 2;                 // current speed index
let playing = true, vt = 0, last = null;
const DISPLAY_T = 4.0;       // seconds of wall-clock to show one crossing

const SW = 900, SH = 340, MX = 55, Y0 = 120;
const ppm = (SW - 2 * MX) / M.L;
const sx = (x) => MX + x * ppm;
const sy = (w) => Y0 + w * M.mag * ppm;
const NS = "http://www.w3.org/2000/svg";
const el = (t, a = {}) => { const e = document.createElementNS(NS, t);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

// ---- scene scaffold --------------------------------------------------------
const scene = document.getElementById("scene");
const support = (x) => { const g = el("g"), xs = sx(x);
  g.appendChild(el("polygon", { points: `${xs},${Y0} ${xs-11},${Y0+18} ${xs+11},${Y0+18}`,
    fill: "#3a4754", stroke: "#5b6b7a" })); return g; };
scene.appendChild(el("line", { x1: sx(0), y1: Y0, x2: sx(M.L), y2: Y0,
  stroke: "#33404c", "stroke-width": 1, "stroke-dasharray": "4 4" }));
scene.appendChild(support(0)); scene.appendChild(support(M.L));
// span dimension label
const dimY = Y0 + 40;
scene.appendChild(el("line", { x1: sx(0), y1: dimY, x2: sx(M.L), y2: dimY,
  stroke: "#5b6b7a", "stroke-width": 1 }));
const dimT = el("text", { x: sx(M.L / 2), y: dimY + 14, fill: "#8b98a5",
  "font-size": 11, "text-anchor": "middle" });
dimT.textContent = `span L = ${M.L} m`; scene.appendChild(dimT);
const deckG = el("g"); scene.appendChild(deckG);
const carG = el("g"); scene.appendChild(carG);

function drawDeck(w) {
  deckG.replaceChildren();
  let p = "";
  for (let i = 0; i < NX.length; i++) p += `${sx(NX[i])},${sy(w[i])} `;
  deckG.appendChild(el("polyline", { points: p.trim(), fill: "none",
    stroke: "#4fc3f7", "stroke-width": 5, "stroke-linecap": "round",
    "stroke-linejoin": "round" }));
  for (let i = 0; i < NX.length; i++)
    deckG.appendChild(el("circle", { cx: sx(NX[i]), cy: sy(w[i]), r: 2, fill: "#dfe7ee" }));
}
function spring(x, y1, y2, c = 5, wd = 6) {
  let p = `${x},${y1}`; const s = (y2 - y1) / (c * 2);
  for (let i = 1; i < c * 2; i++) p += ` ${x + (i % 2 ? wd : -wd)},${y1 + s * i}`;
  return el("polyline", { points: p + ` ${x},${y2}`, fill: "none",
    stroke: "#9fb3c4", "stroke-width": 1.6 });
}
function drawCar(fr) {
  carG.replaceChildren();
  const cx = sx(fr.xv), deckY = sy(fr.wc), R = 9, wT = deckY - 2 * R, Ls = 22, bH = 22, bW = 70;
  const bBot = wT - Ls, bTop = bBot - bH;
  carG.appendChild(el("circle", { cx, cy: deckY - R, r: R, fill: "#222b33",
    stroke: "#aab7c4", "stroke-width": 2 }));
  carG.appendChild(spring(cx - 12, bBot, wT));
  carG.appendChild(el("line", { x1: cx + 12, y1: bBot, x2: cx + 12, y2: wT,
    stroke: "#9fb3c4", "stroke-width": 2.2 }));
  carG.appendChild(el("rect", { x: cx - bW / 2, y: bTop, width: bW, height: bH,
    rx: 4, fill: "#2f6f8f", stroke: "#7fc7e6", "stroke-width": 1.4 }));
  const t = el("text", { x: cx, y: bTop + bH / 2 + 4, fill: "#dff1fb",
    "font-size": 11, "text-anchor": "middle" });
  t.textContent = (M.weight / 1e3).toFixed(0) + " kN"; carG.appendChild(t);
}

// ---- DAF-vs-speed plot -----------------------------------------------------
const dp = document.getElementById("dafPlot");
const PW = 560, PH = 210, mL = 46, mR = 14, mT = 14, mB = 30;
const sp = D.speeds, dc = D.daf_curve;
const vmin = sp[0], vmax = sp[sp.length - 1];
let dmin = Math.min(...dc), dmax = Math.max(...dc);
dmin = Math.min(1.0, dmin) - 0.02; dmax += 0.03;
const dx = (v) => mL + (v - vmin) / (vmax - vmin) * (PW - mL - mR);
const dy = (d) => mT + (dmax - d) / (dmax - dmin) * (PH - mT - mB);
let dafMarker;
function buildDaf() {
  dp.replaceChildren();
  dp.appendChild(el("line", { x1: mL, y1: mT, x2: mL, y2: PH - mB, stroke: "#4a5965" }));
  dp.appendChild(el("line", { x1: mL, y1: PH - mB, x2: PW - mR, y2: PH - mB, stroke: "#4a5965" }));
  // DAF = 1 reference
  dp.appendChild(el("line", { x1: mL, y1: dy(1), x2: PW - mR, y2: dy(1),
    stroke: "#3a4754", "stroke-dasharray": "4 4" }));
  const one = el("text", { x: mL - 5, y: dy(1) + 3, fill: "#8b98a5", "font-size": 9,
    "text-anchor": "end" }); one.textContent = "1.0"; dp.appendChild(one);
  // axis labels
  [vmin, vmax].forEach((v) => { const t = el("text", { x: dx(v), y: PH - mB + 14,
    fill: "#8b98a5", "font-size": 9, "text-anchor": "middle" });
    t.textContent = v; dp.appendChild(t); });
  const xl = el("text", { x: (PW + mL) / 2, y: PH - 4, fill: "#8b98a5",
    "font-size": 10, "text-anchor": "middle" }); xl.textContent = "speed [m/s]";
  dp.appendChild(xl);
  let p = ""; for (let i = 0; i < sp.length; i++) p += `${dx(sp[i])},${dy(dc[i])} `;
  dp.appendChild(el("polyline", { points: p.trim(), fill: "none", stroke: "#4fc3f7",
    "stroke-width": 2 }));
  for (let i = 0; i < sp.length; i++)
    dp.appendChild(el("circle", { cx: dx(sp[i]), cy: dy(dc[i]), r: 2.5, fill: "#4fc3f7" }));
  dafMarker = el("circle", { r: 6, fill: "none", stroke: "#ffd166", "stroke-width": 2.5 });
  dp.appendChild(dafMarker);
}
function moveDaf(i) {
  dafMarker.setAttribute("cx", dx(sp[i])); dafMarker.setAttribute("cy", dy(dc[i]));
}

// ---- readout ---------------------------------------------------------------
const readout = document.getElementById("readout");
function row(a, b, c = "") { return `<div class="row"><span>${a}</span><span class="${c}">${b}</span></div>`; }
function updateReadout(s) {
  readout.innerHTML =
    row("speed", s.speed + " m/s (" + (s.speed * 3.6).toFixed(0) + " km/h)") +
    row("DAF", s.daf.toFixed(3), "hi") +
    row("peak deflection", s.peak_mid_mm.toFixed(2) + " mm") +
    row("static deflection", (M.static_midspan * 1e3).toFixed(2) + " mm") +
    row("bridge f₁", M.f1.toFixed(2) + " Hz") +
    row("damping ζ", M.damping_pct.toFixed(0) + " %") +
    `<div class="group">` +
    row("B-WIM true", (M.weight / 1e3).toFixed(1) + " kN") +
    row("B-WIM recovered", (s.w_rec / 1e3).toFixed(1) + " kN", "hi") +
    `</div>` +
    `<div class="note">DAF stays near 1 because a 40 m bridge is stiff relative to
     traffic speeds — its critical speed is far higher.</div>`;
}

// ---- animation -------------------------------------------------------------
function render() {
  const s = SCN[cur];
  const i = Math.min(s.frames.length - 1, Math.round(vt / DISPLAY_T * (s.frames.length - 1)));
  drawDeck(s.frames[i].w); drawCar(s.frames[i]);
}
function loop(now) {
  if (last === null) last = now;
  vt += (now - last) / 1000; last = now;
  if (vt > DISPLAY_T) vt = 0;
  if (playing) render();
  requestAnimationFrame(loop);
}

// ---- wire up ---------------------------------------------------------------
document.getElementById("magLabel").textContent = "×" + Math.round(M.mag);
document.getElementById("zetaLabel").textContent = M.damping_pct.toFixed(0) + "%";
buildDaf();
function select(i) {
  cur = i; const s = SCN[i];
  document.getElementById("speedLabel").textContent = s.speed + " m/s";
  moveDaf(i); updateReadout(s); render();
}
const playBtn = document.getElementById("playBtn");
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "⏸ Pause" : "▶ Play"; };
const sl = document.getElementById("speed");
sl.max = SCN.length - 1; sl.value = cur;
sl.oninput = (e) => select(parseInt(e.target.value, 10));

select(cur);
requestAnimationFrame(loop);
