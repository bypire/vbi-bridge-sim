/* Drive-by bridge-inspection viewer. Slide the damage severity and watch the
 * contact-point spectrum's bridge peak shift left while the health panel flips.
 * Vanilla JS + inline SVG; data injected via output/driveby_data.js. */

const D = DRIVEBY_DATA;
const M = D.meta;
const NX = D.node_x;
const SPEC_F = D.spec_freq;
const SCN = D.scenarios;

let cur = 0;            // current scenario (severity) index
let playing = true;
let vt = 0, last = null;

// ---- scene geometry --------------------------------------------------------
const SW = 900, SH = 360, MX = 55, Y0 = 125;
const ppm = (SW - 2 * MX) / M.L;
const sx = (x) => MX + x * ppm;
const sy = (w) => Y0 + w * M.mag * ppm;
const SVGNS = "http://www.w3.org/2000/svg";
const el = (tag, a = {}) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in a) e.setAttribute(k, a[k]);
  return e;
};

const scene = document.getElementById("scene");
const damaged = new Set(M.damaged_elems);

function support(x) {
  const g = el("g"), xs = sx(x), ys = Y0;
  g.appendChild(el("polygon", { points: `${xs},${ys} ${xs-11},${ys+18} ${xs+11},${ys+18}`,
    fill: "#3a4754", stroke: "#5b6b7a" }));
  return g;
}
// static scaffold
scene.appendChild(el("line", { x1: sx(0), y1: Y0, x2: sx(M.L), y2: Y0,
  stroke: "#33404c", "stroke-width": 1, "stroke-dasharray": "4 4" }));
scene.appendChild(support(0));
scene.appendChild(support(M.L));
const deckG = el("g"); scene.appendChild(deckG);
const carG = el("g"); scene.appendChild(carG);
// damage-zone label
const zlab = el("text", { x: sx(0.5 * (M.zone[0] + M.zone[1])), y: Y0 + 52,
  fill: "#e0564f", "font-size": 11, "text-anchor": "middle" });
scene.appendChild(zlab);

function drawDeck(w) {
  deckG.replaceChildren();
  for (let i = 0; i < NX.length - 1; i++) {
    const dmg = damaged.has(i);
    deckG.appendChild(el("line", { x1: sx(NX[i]), y1: sy(w[i]),
      x2: sx(NX[i + 1]), y2: sy(w[i + 1]),
      stroke: dmg ? "#e0564f" : "#4fc3f7",
      "stroke-width": dmg ? 7 : 5, "stroke-linecap": "round" }));
  }
  for (let i = 0; i < NX.length; i++)
    deckG.appendChild(el("circle", { cx: sx(NX[i]), cy: sy(w[i]), r: 2, fill: "#dfe7ee" }));
}

function spring(x, y1, y2, coils = 5, wdt = 6) {
  let p = `${x},${y1}`; const s = (y2 - y1) / (coils * 2);
  for (let i = 1; i < coils * 2; i++) p += ` ${x + (i % 2 ? wdt : -wdt)},${y1 + s * i}`;
  return el("polyline", { points: p + ` ${x},${y2}`, fill: "none",
    stroke: "#9fb3c4", "stroke-width": 1.6 });
}

function drawCar(fr) {
  carG.replaceChildren();
  const cx = sx(fr.xv), deckY = sy(fr.wc), R = 9;
  const wheelCy = deckY - R, wheelTop = wheelCy - R, Ls = 22, bH = 22, bW = 64;
  const susp = (fr.body - fr.axle) * M.mag * ppm;
  const bBot = wheelTop - Ls + susp, bTop = bBot - bH;
  carG.appendChild(el("circle", { cx, cy: wheelCy, r: R, fill: "#222b33",
    stroke: "#aab7c4", "stroke-width": 2 }));
  carG.appendChild(spring(cx - 11, bBot, wheelTop));
  carG.appendChild(el("line", { x1: cx + 11, y1: bBot, x2: cx + 11, y2: wheelTop,
    stroke: "#9fb3c4", "stroke-width": 2.2 }));
  carG.appendChild(el("rect", { x: cx - bW / 2, y: bTop, width: bW, height: bH,
    rx: 4, fill: "#2f6f8f", stroke: "#7fc7e6", "stroke-width": 1.4 }));
  // a little accelerometer marker on the body
  carG.appendChild(el("circle", { cx, cy: bTop + 5, r: 2.5, fill: "#ffd166" }));
}

// ---- spectrum panel --------------------------------------------------------
const specSvg = document.getElementById("spectrum");
const SPW = 560, SPH = 240, ML = 44, MR = 12, MT = 14, MB = 30;
const fmin = SPEC_F[0], fmax = SPEC_F[SPEC_F.length - 1];
const px = (f) => ML + (f - fmin) / (fmax - fmin) * (SPW - ML - MR);
const py = (a) => MT + (1 - a) * (SPH - MT - MB);

function polyline(freq, amp, attrs) {
  let p = "";
  for (let i = 0; i < freq.length; i++) p += `${px(freq[i])},${py(amp[i])} `;
  return el("polyline", Object.assign({ points: p.trim(), fill: "none" }, attrs));
}

let specStatic = null;
function buildSpectrumStatic() {
  specSvg.replaceChildren();
  // axes
  specSvg.appendChild(el("line", { x1: ML, y1: MT, x2: ML, y2: SPH - MB,
    stroke: "#4a5965" }));
  specSvg.appendChild(el("line", { x1: ML, y1: SPH - MB, x2: SPW - MR, y2: SPH - MB,
    stroke: "#4a5965" }));
  for (let f = Math.ceil(fmin); f <= fmax; f++) {
    specSvg.appendChild(el("line", { x1: px(f), y1: SPH - MB, x2: px(f), y2: SPH - MB + 4,
      stroke: "#4a5965" }));
    const t = el("text", { x: px(f), y: SPH - MB + 16, fill: "#8b98a5",
      "font-size": 9, "text-anchor": "middle" }); t.textContent = f;
    specSvg.appendChild(t);
  }
  const xl = el("text", { x: (SPW + ML) / 2, y: SPH - 4, fill: "#8b98a5",
    "font-size": 10, "text-anchor": "middle" }); xl.textContent = "frequency [Hz]";
  specSvg.appendChild(xl);
  // healthy baseline frequency line (green)
  specSvg.appendChild(el("line", { x1: px(M.f_healthy), y1: MT, x2: px(M.f_healthy),
    y2: SPH - MB, stroke: "#46c46a", "stroke-width": 1.4, "stroke-dasharray": "5 4" }));
  const gl = el("text", { x: px(M.f_healthy) + 3, y: MT + 10, fill: "#46c46a",
    "font-size": 9 }); gl.textContent = "healthy"; specSvg.appendChild(gl);
  // faint healthy spectrum for comparison
  specSvg.appendChild(polyline(SPEC_F, SCN[0].spectrum,
    { stroke: "#46c46a", "stroke-width": 1, opacity: 0.35 }));
  // dynamic elements
  specStatic = {
    curve: polyline(SPEC_F, SCN[0].spectrum, { stroke: "#e0564f", "stroke-width": 2 }),
    mark: el("line", { x1: px(SCN[0].f_detected), y1: MT,
      x2: px(SCN[0].f_detected), y2: SPH - MB, stroke: "#e0564f", "stroke-width": 1.6 }),
  };
  specSvg.appendChild(specStatic.curve);
  specSvg.appendChild(specStatic.mark);
}

function updateSpectrum(s) {
  let p = "";
  for (let i = 0; i < SPEC_F.length; i++) p += `${px(SPEC_F[i])},${py(s.spectrum[i])} `;
  specStatic.curve.setAttribute("points", p.trim());
  specStatic.mark.setAttribute("x1", px(s.f_detected));
  specStatic.mark.setAttribute("x2", px(s.f_detected));
}

// ---- health panel ----------------------------------------------------------
const health = document.getElementById("health");
function updateHealth(s) {
  const damaged = s.drop_pct > 2.0;        // ~ one resolution bin
  const status = damaged
    ? `<span style="color:#e0564f">⚠ DAMAGE DETECTED</span>`
    : `<span style="color:#46c46a">✔ HEALTHY</span>`;
  const row = (a, b, c = "") =>
    `<div class="row"><span>${a}</span><span class="${c}">${b}</span></div>`;
  health.innerHTML =
    row("status", status) +
    row("healthy baseline", M.f_healthy.toFixed(2) + " Hz") +
    row("detected freq", s.f_detected.toFixed(2) + " Hz", "hi") +
    row("frequency drop", s.drop_pct.toFixed(1) + " %") +
    `<div class="group">` +
    row("damage severity", (s.severity * 100).toFixed(0) + " % EI loss") +
    row("scan speed", M.v_kmh.toFixed(0) + " km/h") +
    `</div>`;
}

// ---- animation -------------------------------------------------------------
function frameIdx(t, frames) {
  const tEnd = frames[frames.length - 1].t;
  return Math.min(frames.length - 1, Math.round(t / tEnd * (frames.length - 1)));
}
function render() {
  const s = SCN[cur];
  const fr = s.frames[frameIdx(vt, s.frames)];
  drawDeck(fr.w);
  drawCar(fr);
}
function loop(now) {
  if (last === null) last = now;
  const dt = (now - last) / 1000; last = now;
  if (playing) {
    const tEnd = SCN[cur].frames[SCN[cur].frames.length - 1].t;
    vt += dt; if (vt > tEnd) vt = 0;
    render();
  }
  requestAnimationFrame(loop);
}

// ---- wire up ---------------------------------------------------------------
document.getElementById("magLabel").textContent = "×" + Math.round(M.mag);
zlab.textContent = `damage zone (${M.zone[0]}–${M.zone[1]} m)`;
buildSpectrumStatic();

function selectScenario(i) {
  cur = i;
  const s = SCN[i];
  document.getElementById("sevLabel").textContent = (s.severity * 100).toFixed(0) + "%";
  updateSpectrum(s); updateHealth(s); render();
}

const playBtn = document.getElementById("playBtn");
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "⏸ Pause" : "▶ Play"; };
document.getElementById("sev").oninput = (e) => selectScenario(parseInt(e.target.value, 10));

// optional deep-link for screenshots/inspection: driveby.html#sev=5
const hashSev = new URLSearchParams(location.hash.slice(1)).get("sev");
const startSev = hashSev === null ? 0
  : Math.max(0, Math.min(SCN.length - 1, parseInt(hashSev, 10)));
document.getElementById("sev").value = startSev;
selectScenario(startSev);
requestAnimationFrame(loop);
