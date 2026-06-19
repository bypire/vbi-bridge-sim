/* BridgeTwin ops dashboard. Plays the precomputed traffic stream: each truck is
 * weighed by B-WIM (data from export_traffic.py); we animate the live twin and
 * update KPIs, the weighing feed, a weight-distribution histogram and an
 * accuracy scatter. The bridge deflection in the twin is computed analytically
 * in-browser (simply-supported point-load deflection) so no per-truck frames
 * need to be shipped. Vanilla JS + inline SVG. */

const T = TRAFFIC_DATA;
const META = T.meta;
const VEH = T.vehicles;
const L = META.span_L;

const NS = "http://www.w3.org/2000/svg";
const el = (t, a = {}) => { const e = document.createElementNS(NS, t);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

document.getElementById("bridgeId").textContent = META.bridge_id;

// ---- analytic deflection: simply-supported beam, unit point load at a -------
// shape only (magnified for display); w(x) for load P at position a.
function deflShape(a, P) {
  const n = 40, EI = META.E * META.I, w = [];
  for (let k = 0; k <= n; k++) {
    const x = L * k / n, b = L - a;
    let val;
    if (x <= a) val = P * b * x * (L * L - b * b - x * x) / (6 * L * EI);
    else        val = P * a * (L - x) * (2 * L * x - a * a - x * x) / (6 * L * EI);
    w.push(val);
  }
  return w;
}

// ---- live twin -------------------------------------------------------------
const twin = document.getElementById("twin");
const TW = 560, TH = 240, TMX = 40, TY0 = 95;
const tsx = (x) => TMX + x / L * (TW - 2 * TMX);
// magnify: a fully-loaded ~400 kN truck should dip ~70 px
const MAG = 70 / (Math.max(...VEH.map(v => v.true_gvw)) * 1e3 *
  L ** 3 / (48 * META.E * META.I));
const tsy = (w) => TY0 + w * MAG;
function support(x) { const g = el("g"), xs = tsx(x);
  g.appendChild(el("polygon", { points: `${xs},${TY0} ${xs-9},${TY0+15} ${xs+9},${TY0+15}`,
    fill: "#3a4754", stroke: "#5b6b7a" })); return g; }
const twinDeck = el("g"), twinCar = el("g");
twin.appendChild(el("line", { x1: tsx(0), y1: TY0, x2: tsx(L), y2: TY0,
  stroke: "#33404c", "stroke-dasharray": "4 4" }));
twin.appendChild(support(0)); twin.appendChild(support(L));
twin.appendChild(el("line", { x1: tsx(META.gauge_x), y1: TY0 - 6, x2: tsx(META.gauge_x),
  y2: TY0 + 22, stroke: "#c8a24a", "stroke-dasharray": "2 3" }));
twin.appendChild(twinDeck); twin.appendChild(twinCar);

function drawTwin(a, gvwN, cls, overload) {
  const w = deflShape(a, gvwN);
  let p = ""; for (let k = 0; k < w.length; k++) p += `${tsx(L*k/(w.length-1))},${tsy(w[k])} `;
  twinDeck.replaceChildren(el("polyline", { points: p.trim(), fill: "none",
    stroke: "#4fc3f7", "stroke-width": 4, "stroke-linejoin": "round" }));
  twinCar.replaceChildren();
  if (a < 0 || a > L) return;
  const cx = tsx(a), cy = tsy(deflAt(w, a)) - 16, col = overload ? "#e0564f" : "#2f6f8f";
  twinCar.appendChild(el("rect", { x: cx - 22, y: cy - 14, width: 44, height: 16, rx: 3,
    fill: col, stroke: "#7fc7e6", "stroke-width": 1.2 }));
  twinCar.appendChild(el("circle", { cx: cx - 12, cy: cy + 4, r: 4, fill: "#222b33", stroke: "#aab7c4" }));
  twinCar.appendChild(el("circle", { cx: cx + 12, cy: cy + 4, r: 4, fill: "#222b33", stroke: "#aab7c4" }));
}
function deflAt(w, a) { const f = a / L * (w.length - 1), i = Math.floor(f);
  return i >= w.length - 1 ? w[w.length-1] : w[i] + (w[i+1]-w[i]) * (f-i); }

// ---- KPI cards -------------------------------------------------------------
function renderKpis(seen) {
  const n = seen.length;
  const over = seen.filter(v => v.overload).length;
  const flagged = seen.filter(v => v.flagged).length;
  const correct = seen.filter(v => v.overload === v.flagged).length;
  const heaviest = n ? Math.max(...seen.map(v => v.true_gvw)) : 0;
  const mae = n ? seen.reduce((s, v) => s + Math.abs(v.err_pct), 0) / n : 0;
  const card = (v, l, cls = "") => `<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  document.getElementById("kpis").innerHTML =
    card(n, "trucks weighed") +
    card(flagged, "flagged overweight", flagged ? "warn" : "") +
    card(heaviest.toFixed(0) + " kN", "heaviest GVW") +
    card(mae.toFixed(1) + " %", "mean B-WIM error") +
    card((n ? 100 * correct / n : 100).toFixed(0) + " %", "detection accuracy", "good");
}

// ---- feed ------------------------------------------------------------------
const feed = document.getElementById("feed");
function verdict(v) {
  if (v.uncertain) return ["unc", "UNCERTAIN"];
  if (v.flagged) return ["ov", "OVERWEIGHT"];
  return ["ok", "OK"];
}
function addFeed(v) {
  const row = document.createElement("div");
  const [cls, label] = verdict(v);
  row.className = "feed-row" + (cls === "ov" ? " ov" : cls === "unc" ? " unc" : "");
  const mm = String(Math.floor(v.arrival / 60)).padStart(2, "0");
  const ss = String(Math.floor(v.arrival % 60)).padStart(2, "0");
  row.innerHTML = `<span class="t">${mm}:${ss}</span>` +
    `<span>${v.cls}</span>` +
    `<span class="w">${v.rec_gvw.toFixed(0)}±${v.ci_kN.toFixed(0)} kN</span>` +
    `<span class="badge ${cls}">${label}</span>`;
  feed.prepend(row);
  while (feed.children.length > 9) feed.removeChild(feed.lastChild);
}

// ---- histogram (GVW / legal ratio) -----------------------------------------
function buildHist(seen) {
  const svg = document.getElementById("hist"); svg.replaceChildren();
  const W = 560, H = 220, mL = 36, mR = 12, mT = 12, mB = 30;
  const ratios = seen.map(v => v.rec_gvw / v.legal);
  const lo = 0.3, hi = 1.7, nb = 28;
  const bins = new Array(nb).fill(0);
  ratios.forEach(r => { const b = Math.min(nb-1, Math.max(0, Math.floor((r-lo)/(hi-lo)*nb))); bins[b]++; });
  const maxc = Math.max(1, ...bins);
  const bx = (i) => mL + i / nb * (W - mL - mR);
  const by = (c) => mT + (1 - c / maxc) * (H - mT - mB);
  for (let i = 0; i < nb; i++) {
    const r0 = lo + i / nb * (hi - lo);
    svg.appendChild(el("rect", { x: bx(i) + 1, y: by(bins[i]),
      width: (W-mL-mR)/nb - 2, height: H - mB - by(bins[i]),
      fill: r0 >= 1.0 ? "#e0564f" : "#4fc3f7", opacity: 0.85 }));
  }
  const xLimit = mL + (1.0 - lo) / (hi - lo) * (W - mL - mR);
  svg.appendChild(el("line", { x1: xLimit, y1: mT, x2: xLimit, y2: H - mB,
    stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
  const lab = el("text", { x: xLimit + 4, y: mT + 12, fill: "#fff", "font-size": 10 });
  lab.textContent = "legal limit"; svg.appendChild(lab);
  [[lo,"0.3"],[1.0,"1.0"],[hi,"1.7"]].forEach(([r,t]) => {
    const x = mL + (r-lo)/(hi-lo)*(W-mL-mR);
    const e = el("text", { x, y: H-10, fill: "#8b98a5", "font-size": 9, "text-anchor": "middle" });
    e.textContent = t; svg.appendChild(e); });
  const xl = el("text", { x: (W+mL)/2, y: H-22, fill: "#8b98a5", "font-size": 9, "text-anchor": "middle", opacity: 0 });
  svg.appendChild(xl);
}

// ---- accuracy scatter ------------------------------------------------------
function buildScatter(seen) {
  const svg = document.getElementById("scatter"); svg.replaceChildren();
  const W = 560, H = 220, m = 34;
  const all = VEH.map(v => v.true_gvw).concat(VEH.map(v => v.rec_gvw));
  const lo = 0, hi = Math.max(...all) * 1.05;
  const sx = (v) => m + v / hi * (W - 2 * m);
  const sy = (v) => H - m - v / hi * (H - 2 * m);
  // axes + 1:1 line
  svg.appendChild(el("line", { x1: m, y1: H-m, x2: W-m, y2: H-m, stroke: "#4a5965" }));
  svg.appendChild(el("line", { x1: m, y1: m, x2: m, y2: H-m, stroke: "#4a5965" }));
  svg.appendChild(el("line", { x1: sx(0), y1: sy(0), x2: sx(hi), y2: sy(hi),
    stroke: "#5b6b7a", "stroke-dasharray": "4 4" }));
  seen.forEach(v => {
    const c = v.overload ? "#e0564f" : "#4fc3f7";
    // 95% CI error bar on the recovered weight
    svg.appendChild(el("line", { x1: sx(v.true_gvw), y1: sy(v.rec_gvw - v.ci_kN),
      x2: sx(v.true_gvw), y2: sy(v.rec_gvw + v.ci_kN), stroke: c, opacity: 0.35 }));
    svg.appendChild(el("circle", { cx: sx(v.true_gvw), cy: sy(v.rec_gvw),
      r: 3, fill: c, opacity: 0.9 }));
  });
  const xl = el("text", { x: (W)/2, y: H-8, fill: "#8b98a5", "font-size": 9, "text-anchor": "middle" });
  xl.textContent = "true GVW [kN]"; svg.appendChild(xl);
  const yl = el("text", { x: 10, y: H/2, fill: "#8b98a5", "font-size": 9,
    transform: `rotate(-90 10 ${H/2})`, "text-anchor": "middle" });
  yl.textContent = "recovered GVW [kN]"; svg.appendChild(yl);
}

// ---- now reading -----------------------------------------------------------
function setNow(v) {
  const [cls, label] = verdict(v);
  const col = cls === "ov" ? "ov" : cls === "unc" ? "unc" : "ok";
  document.getElementById("nowReading").innerHTML =
    `weighing <strong>${v.cls}</strong> @ ${v.speed_kmh.toFixed(0)} km/h → ` +
    `<strong>${v.rec_gvw.toFixed(0)} ± ${v.ci_kN.toFixed(0)} kN</strong> ` +
    `· P(overload) <strong>${(100*v.p_over).toFixed(0)}%</strong> ` +
    `<span class="${col}">${cls === "ov" ? "⚠" : cls === "unc" ? "?" : "✓"} ${label}</span>`;
}

// ---- playback --------------------------------------------------------------
let idx = 0, seen = [], crossing = null, crossT = 0;
const CROSS_DUR = 1.4;       // s to cross the twin
const GAP = 1.1;             // s between trucks (compressed timeline)
let gapT = 0, lastNow = null;

renderKpis([]); buildHist([]); buildScatter([]);

// screenshot/preview helper: dashboard.html#full pre-populates everything
if (location.hash.includes("full")) {
  seen = VEH.slice();
  seen.forEach(addFeed);
  renderKpis(seen); buildHist(seen); buildScatter(seen);
  idx = VEH.length;  // animation loop will wrap and replay from a clean feed
}

function startNext() {
  if (idx >= VEH.length) { idx = 0; seen = []; feed.replaceChildren(); }  // loop
  crossing = VEH[idx++]; crossT = 0; setNow(crossing);
}
if (location.hash.includes("full")) {     // preview: keep feed, just animate truck 0
  crossing = VEH[0]; crossT = 0; setNow(crossing);
} else {
  startNext();
}

function frame(now) {
  if (lastNow === null) lastNow = now;
  const dt = (now - lastNow) / 1000; lastNow = now;
  if (crossing) {
    crossT += dt;
    const a = crossT / CROSS_DUR * L;
    drawTwin(Math.min(a, L), crossing.true_gvw * 1e3, crossing.cls, crossing.flagged);
    if (crossT >= CROSS_DUR) {
      seen.push(crossing); addFeed(crossing);
      renderKpis(seen); buildHist(seen); buildScatter(seen);
      crossing = null; gapT = 0;
    }
  } else {
    gapT += dt;
    if (gapT >= GAP) startNext();
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
