/* VBI viewer — draws the injected SIM_DATA: a quarter-car crossing a
 * simply-supported beam. Vanilla JS + inline SVG, no framework, no fetch.
 *
 * World units are metres; we map them to screen pixels with one scale (ppm) for
 * both axes, and the deflection is pre-magnified by SIM_DATA.meta.mag (shown in
 * the header) so the millimetric real dip is visible without faking anything. */

const D = SIM_DATA;
const M = D.meta;
const NX = D.node_x;                 // node x-positions [m]
const FR = D.frames;                 // animation frames
const MID = Math.floor(M.n_nodes / 2);

// ---- scene geometry --------------------------------------------------------
const SCENE_W = 900, SCENE_H = 380;
const MX = 55;                        // side margin [px]
const Y0 = 150;                       // undeformed deck level [px]
const ppm = (SCENE_W - 2 * MX) / M.L; // pixels per metre (same for x and y)

const sx = (x) => MX + x * ppm;
const sy = (w) => Y0 + w * M.mag * ppm;   // w real [m] -> magnified screen y

const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

// diverging blue(−) → grey(0) → red(+) colour map for bending moment
function momentColor(v) {
  const t = Math.max(-1, Math.min(1, v / M.M_abs_max)); // -1..1
  const mix = (a, b, f) => Math.round(a + (b - a) * f);
  if (t >= 0) {                                          // grey -> red (sagging)
    return `rgb(${mix(150, 229, t)},${mix(160, 57, t)},${mix(170, 53, t)})`;
  }                                                      // grey -> blue (hogging)
  const f = -t;
  return `rgb(${mix(150, 66, f)},${mix(160, 135, f)},${mix(170, 245, f)})`;
}

// ---- static scaffold (supports, undeformed deck, gauge marker) -------------
const scene = document.getElementById("scene");

function support(x) {
  const g = el("g");
  const xs = sx(x), ys = Y0;
  g.appendChild(el("polygon", {
    points: `${xs},${ys} ${xs - 11},${ys + 18} ${xs + 11},${ys + 18}`,
    fill: "#3a4754", stroke: "#5b6b7a",
  }));
  // hatched ground
  for (let i = -11; i <= 11; i += 5) {
    g.appendChild(el("line", {
      x1: xs + i, y1: ys + 18, x2: xs + i - 5, y2: ys + 25,
      stroke: "#5b6b7a", "stroke-width": 1,
    }));
  }
  return g;
}

scene.appendChild(el("line", {                            // undeformed reference
  x1: sx(0), y1: Y0, x2: sx(M.L), y2: Y0,
  stroke: "#33404c", "stroke-width": 1, "stroke-dasharray": "4 4",
}));
// gauge section marker
scene.appendChild(el("line", {
  x1: sx(M.gauge_x), y1: Y0 - 8, x2: sx(M.gauge_x), y2: Y0 + 30,
  stroke: "#c8a24a", "stroke-width": 1, "stroke-dasharray": "2 3",
}));
scene.appendChild((() => {
  const t = el("text", { x: sx(M.gauge_x) + 4, y: Y0 + 40, fill: "#c8a24a",
    "font-size": 10 });
  t.textContent = "strain gauge";
  return t;
})());
scene.appendChild(support(0));
scene.appendChild(support(M.L));

// dynamic groups (rebuilt each frame)
const deckG = el("g");      scene.appendChild(deckG);
const carG = el("g");       scene.appendChild(carG);

// ---- per-frame drawing -----------------------------------------------------
function drawDeck(w, Mvals) {
  deckG.replaceChildren();
  // coloured segments between nodes
  for (let i = 0; i < NX.length - 1; i++) {
    const mAvg = 0.5 * (Mvals[i] + Mvals[i + 1]);
    deckG.appendChild(el("line", {
      x1: sx(NX[i]), y1: sy(w[i]), x2: sx(NX[i + 1]), y2: sy(w[i + 1]),
      stroke: momentColor(mAvg), "stroke-width": 5, "stroke-linecap": "round",
    }));
  }
  for (let i = 0; i < NX.length; i++) {
    deckG.appendChild(el("circle", { cx: sx(NX[i]), cy: sy(w[i]), r: 2.2,
      fill: "#dfe7ee" }));
  }
}

function spring(x, y1, y2, coils = 5, width = 6) {
  let pts = `${x},${y1}`;
  const seg = (y2 - y1) / (coils * 2);
  for (let i = 1; i < coils * 2; i++) {
    pts += ` ${x + (i % 2 ? width : -width)},${y1 + seg * i}`;
  }
  pts += ` ${x},${y2}`;
  return el("polyline", { points: pts, fill: "none", stroke: "#9fb3c4",
    "stroke-width": 1.6 });
}

function drawCar(fr) {
  carG.replaceChildren();
  const cx = sx(fr.xv);
  const deckY = sy(fr.wc);            // deck surface under the wheel
  const wheelR = 9;
  const wheelCy = deckY - wheelR;
  const Ls = 24;                      // nominal suspension length [px]
  const bodyH = 24, bodyW = 74;
  // The wheel already rides the (magnified) deck. The body sits a fixed height
  // above it, offset only by the SUSPENSION's dynamic travel (body − axle); the
  // shared bridge motion is in wheelCy, so we must not magnify the body's
  // absolute displacement (that would send it off-screen).
  const susp = (fr.body - fr.axle) * M.mag * ppm;  // suspension travel [px]
  const wheelTop = wheelCy - wheelR;
  const bodyBottom = wheelTop - Ls + susp;
  const bodyTop = bodyBottom - bodyH;

  // tyre (line from contact to axle, gives a sense of tyre deflection)
  carG.appendChild(el("circle", { cx, cy: wheelCy, r: wheelR,
    fill: "#222b33", stroke: "#aab7c4", "stroke-width": 2 }));
  carG.appendChild(el("circle", { cx, cy: wheelCy, r: 2, fill: "#aab7c4" }));
  // suspension spring + damper
  carG.appendChild(spring(cx - 12, bodyBottom, wheelTop));
  carG.appendChild(el("line", { x1: cx + 12, y1: bodyBottom,
    x2: cx + 12, y2: wheelTop, stroke: "#9fb3c4", "stroke-width": 2.4 }));
  // body (sprung mass)
  carG.appendChild(el("rect", { x: cx - bodyW / 2, y: bodyTop, width: bodyW,
    height: bodyH, rx: 4, fill: "#2f6f8f", stroke: "#7fc7e6",
    "stroke-width": 1.5 }));
  const lab = el("text", { x: cx, y: bodyTop + bodyH / 2 + 4, fill: "#dff1fb",
    "font-size": 11, "text-anchor": "middle" });
  lab.textContent = (M.weight / 1e3).toFixed(0) + " kN";
  carG.appendChild(lab);
}

// ---- readout panel ---------------------------------------------------------
const readout = document.getElementById("readout");
function row(label, value, cls = "") {
  return `<div class="row"><span>${label}</span><span class="${cls}">${value}</span></div>`;
}
function updateReadout(fr) {
  const wMidMm = fr.w[MID] * 1000;
  readout.innerHTML =
    row("time", fr.t.toFixed(3) + " s") +
    row("wheel position", fr.xv.toFixed(2) + " m") +
    row("speed", M.speed_kmh.toFixed(0) + " km/h") +
    row("mid-span defl.", wMidMm.toFixed(2) + " mm") +
    row("contact force", (fr.fc / 1e3).toFixed(1) + " kN") +
    row("DAF (peak)", M.daf.toFixed(3), "hi") +
    `<div class="group">` +
    row("B-WIM true", (M.W_true / 1e3).toFixed(1) + " kN") +
    row("B-WIM recovered", (M.W_rec / 1e3).toFixed(1) + " kN", "hi") +
    row("recovery error", M.bwim_error_pct.toFixed(2) + " %") +
    `</div>`;
}

// ---- time-series plots -----------------------------------------------------
function makePlot(svgId, xs, ys, unitColor) {
  const svg = document.getElementById(svgId);
  const W = 430, H = 170, ML = 40, MR = 10, MT = 12, MB = 24;
  const xmin = xs[0], xmax = xs[xs.length - 1];
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const pad = 0.08 * (ymax - ymin); ymin -= pad; ymax += pad;
  const px = (x) => ML + (x - xmin) / (xmax - xmin) * (W - ML - MR);
  const py = (y) => MT + (ymax - y) / (ymax - ymin) * (H - MT - MB);

  // zero line if range spans it
  if (ymin < 0 && ymax > 0) {
    svg.appendChild(el("line", { x1: ML, y1: py(0), x2: W - MR, y2: py(0),
      stroke: "#33404c", "stroke-width": 1 }));
  }
  // axes
  svg.appendChild(el("line", { x1: ML, y1: MT, x2: ML, y2: H - MB,
    stroke: "#4a5965", "stroke-width": 1 }));
  svg.appendChild(el("line", { x1: ML, y1: H - MB, x2: W - MR, y2: H - MB,
    stroke: "#4a5965", "stroke-width": 1 }));
  // y labels
  const yl = (v, txt) => {
    const t = el("text", { x: ML - 5, y: py(v) + 3, fill: "#8b98a5",
      "font-size": 9, "text-anchor": "end" });
    t.textContent = txt; svg.appendChild(t);
  };
  yl(ymax, ymax.toFixed(ymax > 100 ? 0 : 1));
  yl(ymin, ymin.toFixed(ymin > 100 ? 0 : 1));
  const xt = el("text", { x: W - MR, y: H - 8, fill: "#8b98a5", "font-size": 9,
    "text-anchor": "end" });
  xt.textContent = xmax.toFixed(2) + " s"; svg.appendChild(xt);

  // data polyline
  let pts = "";
  for (let i = 0; i < xs.length; i++) pts += `${px(xs[i])},${py(ys[i])} `;
  svg.appendChild(el("polyline", { points: pts.trim(), fill: "none",
    stroke: unitColor, "stroke-width": 1.6 }));

  // moving marker
  const marker = el("line", { x1: px(xmin), y1: MT, x2: px(xmin), y2: H - MB,
    stroke: "#e6edf3", "stroke-width": 1, "stroke-dasharray": "3 3" });
  svg.appendChild(marker);
  return (t) => { const X = px(t); marker.setAttribute("x1", X);
    marker.setAttribute("x2", X); };
}

const markDefl = makePlot("plotDefl", D.series.t,
  D.series.w_mid.map((v) => v * 1000), "#4fc3f7");
const markForce = makePlot("plotForce", D.series.t,
  D.series.fc.map((v) => v / 1000), "#7fdca0");

// ---- header labels ---------------------------------------------------------
const peakMm = Math.max(...FR.map((f) => Math.abs(f.w[MID]))) * 1000;
document.getElementById("magLabel").textContent = "×" + Math.round(M.mag);
document.getElementById("peakLabel").textContent = peakMm.toFixed(1) + " mm";

// ---- animation loop --------------------------------------------------------
const tEnd = FR[FR.length - 1].t;
let vt = 0;               // virtual time [s]
let playing = true;
let last = null;
let speed = parseFloat(document.getElementById("speed").value);

const scrub = document.getElementById("scrub");
scrub.max = FR.length - 1;

function frameIndexForTime(t) {
  return Math.min(FR.length - 1, Math.round(t / tEnd * (FR.length - 1)));
}

function render(idx) {
  const fr = FR[idx];
  drawDeck(fr.w, fr.M);
  drawCar(fr);
  updateReadout(fr);
  markDefl(fr.t);
  markForce(fr.t);
  scrub.value = idx;
}

function loop(now) {
  if (last === null) last = now;
  const dt = (now - last) / 1000;
  last = now;
  if (playing) {
    vt += dt * speed;
    if (vt > tEnd) vt = 0;            // loop the crossing
    render(frameIndexForTime(vt));
  }
  requestAnimationFrame(loop);
}

// ---- controls --------------------------------------------------------------
const playBtn = document.getElementById("playBtn");
playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
};
document.getElementById("restartBtn").onclick = () => { vt = 0; };
document.getElementById("speed").oninput = (e) => {
  speed = parseFloat(e.target.value);
  document.getElementById("speedLabel").textContent = speed.toFixed(1) + "×";
};
scrub.oninput = (e) => {
  playing = false; playBtn.textContent = "▶ Play";
  const idx = parseInt(e.target.value, 10);
  vt = idx / (FR.length - 1) * tEnd;
  render(idx);
};

// Optional deep-link for development: index.html#f=200 starts paused on a given
// frame (handy for screenshots / inspecting a specific instant).
const hash = new URLSearchParams(location.hash.slice(1));
let startIdx = 0;
if (hash.has("f")) {
  startIdx = Math.max(0, Math.min(FR.length - 1, parseInt(hash.get("f"), 10)));
  vt = startIdx / (FR.length - 1) * tEnd;
  playing = false;
  playBtn.textContent = "▶ Play";
}

render(startIdx);
requestAnimationFrame(loop);
