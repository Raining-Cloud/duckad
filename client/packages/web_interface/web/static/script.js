"use strict";

/* ===========================================================================
   Shared drawing helpers
   =========================================================================== */
const CSS = getComputedStyle(document.documentElement);
const C = (name) => CSS.getPropertyValue(name).trim();

/* The model was trained in CARLA, whose world is 20x Duckietown scale, so every
   distance it emits is in CARLA metres. controller_node divides it out once, in
   _parse_trajectory, before republishing active_spatial / active_temporal — so
   the trajectory, the camera overlay, speed_bins_kmh and commanded_speed_kmh
   all arrive in ROBOT units and must NOT be scaled again here.
   The one exception is /model_node/speed_kmh: onnx_model.py computes it from
   the RAW tensor, so it lands 20x too high and is the only field this page has
   to correct. (Proof: it is the same norm as speed_bins[0], which the
   controller takes after dividing — the two differ by exactly this factor.) */
const CARLA_SCALE = 20.0;
const KMH_PER_MPS = 3.6;

/* Model waypoints are 1 m apart in CARLA units, so the real corridor is
   1/20 = 5 cm per waypoint — roughly half a metre of reach over 10 points.
   That is why this page works in centimetres and m/s, not metres and km/h:
   at Duckiebot scale the km/h figures are all under 1 and read as noise. */
const MODEL_WAYPOINT_SPACING_M = 1.0;
const WAYPOINT_SPACING_M = MODEL_WAYPOINT_SPACING_M / CARLA_SCALE;   // 0.05 m

/* /model_node/speed_kmh is CARLA km/h off the RAW tensor: two corrections.
     |temporal[0]| / dt / CARLA_SCALE  =  1.0 / 0.25 / 20  =  0.2 m/s        */
const modelSpeedMps = (st) => st.speed_kmh / KMH_PER_MPS / CARLA_SCALE;

/* controller_node topics are already robot-frame (it divides in
   _parse_trajectory) — they only need km/h -> m/s. */
const kmhToMps = (v) => v / KMH_PER_MPS;

const COMMAND_NAMES  = ["default", "left", "straight", "right"];
const COMMAND_COLORS = [C("--cmd-default"), C("--cmd-left"), C("--cmd-straight"), C("--cmd-right")];
const SPATIAL  = C("--spatial");
const TEMPORAL = C("--temporal");
const GOAL     = C("--goal");
const RULE     = C("--rule");
const RULE_SOFT= C("--rule-soft");
const TEXT     = C("--text");
const TEXT_MID = C("--text-mid");
const TEXT_DIM = C("--text-dim");
const WELL     = C("--ink-well");
/* Canvas takes the same stack as the CSS, read from the token, so the charts
   can never drift onto a different face than the surrounding numerals. */
const FONT_DATA = C("--font-data").replace(/\s+/g, " ")
              || '"DejaVu Sans Mono", monospace';

/* Size the bitmap to the CSS box at device resolution: a fixed bitmap the
   browser then rescales is what makes labels and hairlines look soft. */
function fitCanvas(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = cv.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

function label(ctx, text, x, y, { size = 11, color = TEXT_DIM, align = "center", weight = 400 } = {}) {
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px ${FONT_DATA}`;
  ctx.textAlign = align;
  ctx.textBaseline = "alphabetic";
  ctx.fillText(text, x, y);
}

function roundRectPath(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/* ===========================================================================
   1. BIRD'S-EYE TRAJECTORY
   Plan view of the predicted path with distance rings, a lane-width reference
   and the goal-point chord. Isometric, so both axes share one scale.
   =========================================================================== */
const trajCv = document.getElementById("trajectory");

function drawTrajectory(st) {
  const fit = fitCanvas(trajCv); if (!fit) return;
  const { ctx, w, h } = fit;
  ctx.clearRect(0, 0, w, h);

  /* Isometric, so one scale governs both axes — but the corridor is long and
     narrow (metres forward, centimetres sideways), so a bare max() would
     collapse the lateral axis to nothing. Floor each extent separately and let
     forward bind the scale. */
  let maxX = 6 * WAYPOINT_SPACING_M, maxY = 2 * WAYPOINT_SPACING_M;
  for (const [x, y] of st.spatial.concat(st.temporal)) {
    maxX = Math.max(maxX, Math.abs(x));
    maxY = Math.max(maxY, Math.abs(y));
  }
  const rangeX = maxX * 1.12, rangeY = maxY * 1.35;
  const range = rangeX;

  const cx = w / 2, cy = h - 26;
  const scale = Math.min((w / 2) / rangeY, (h - 40) / rangeX);
  const toPx = (x, y) => [cx - y * scale, cy - x * scale];

  /* Distance rings — a plan view without a scale is a decoration. At robot
     scale the whole corridor is well under a metre, so the rings step in
     centimetres and are labelled as such; metres here would read "0.1, 0.2". */
  const step = range > 6 ? 2 : range > 3 ? 1 : range > 1.2 ? 0.5
             : range > 0.6 ? 0.2 : range > 0.25 ? 0.1 : 0.05;
  const asCm = range <= 1.2;
  ctx.lineWidth = 1;
  for (let d = step; d <= range + 1e-9; d += step) {
    ctx.strokeStyle = RULE_SOFT;
    ctx.beginPath();
    ctx.arc(cx, cy, d * scale, Math.PI, 2 * Math.PI);
    ctx.stroke();
    // Right-edge tick column, not above the ring apex: the apex is exactly where
    // the path runs, so labels there sat on top of the waypoints.
    const text = asCm ? (d * 100).toFixed(0) + " cm" : d.toFixed(step < 1 ? 1 : 0) + " m";
    label(ctx, text, w - 4, cy - d * scale - 3, { align: "right", size: 10.5 });
  }

  /* lane reference: a Duckietown lane is ~0.22 m half-width from centre */
  ctx.strokeStyle = RULE_SOFT;
  ctx.setLineDash([2, 5]);
  for (const off of [-0.22, 0.22]) {
    const [px] = toPx(0, off);
    ctx.beginPath(); ctx.moveTo(px, cy); ctx.lineTo(px, 8); ctx.stroke();
  }
  ctx.setLineDash([]);

  /* centre axis */
  ctx.strokeStyle = RULE;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, 8); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(10, cy); ctx.lineTo(w - 10, cy); ctx.stroke();

  /* The path keeps its own hue. Tinting it by command put it on --cmd-straight,
     which is --temporal's colour: two model outputs in one hue. The command is
     carried by the chevron ring below and the status chips. */
  const spatialColor = SPATIAL;

  /* temporal points (speed samples) sit under the path */
  ctx.fillStyle = TEMPORAL;
  for (const [x, y] of st.temporal) {
    const [px, py] = toPx(x, y);
    ctx.beginPath(); ctx.arc(px, py, 3, 0, 6.2832); ctx.fill();
  }

  /* goal-point chord + apex */
  if (st.goal_indices.length === 3) {
    const [s, a, e] = st.goal_indices;
    if ([s, a, e].every((i) => i >= 0 && i < st.spatial.length)) {
      const [sx, sy] = toPx(...st.spatial[s]);
      const [ex, ey] = toPx(...st.spatial[e]);
      ctx.strokeStyle = TEXT_DIM;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  /* spatial path — line + nodes, matching the camera overlay */
  if (st.spatial.length) {
    ctx.strokeStyle = spatialColor;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    for (const [x, y] of st.spatial) { const [px, py] = toPx(x, y); ctx.lineTo(px, py); }
    ctx.stroke();

    ctx.fillStyle = spatialColor;
    for (const [x, y] of st.spatial) {
      const [px, py] = toPx(x, y);
      ctx.beginPath(); ctx.arc(px, py, 4, 0, 6.2832); ctx.fill();
    }
  }

  if (st.goal_indices.length === 3) {
    const apex = st.spatial[st.goal_indices[1]];
    if (apex) {
      const [px, py] = toPx(...apex);
      ctx.strokeStyle = GOAL; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, 8, 0, 6.2832); ctx.stroke();
      ctx.fillStyle = GOAL;
      ctx.beginPath(); ctx.arc(px, py, 3.5, 0, 6.2832); ctx.fill();
    }
  }

  /* the robot, as a heading chevron rather than an anonymous dot, ringed in
     the active command's colour — that is where the command now lives */
  ctx.strokeStyle = COMMAND_COLORS[st.active_action] ?? TEXT;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy - 1, 12, 0, 6.2832); ctx.stroke();

  ctx.fillStyle = TEXT;
  ctx.beginPath();
  ctx.moveTo(cx, cy - 9);
  ctx.lineTo(cx + 6.5, cy + 5);
  ctx.lineTo(cx, cy + 1.5);
  ctx.lineTo(cx - 6.5, cy + 5);
  ctx.closePath();
  ctx.fill();

  /* The slack an isometric plan view leaves at the sides carries the numbers
     you reach for when tuning the goal point: where it actually landed, and
     the heading error the lateral PID is being asked to close. */
  if (st.goal_indices.length === 3) {
    const apex = st.spatial[st.goal_indices[1]];
    if (apex) {
      const [gx, gy] = apex;
      const headingDeg = Math.atan2(gy, gx) * 180 / Math.PI;
      const rows = [
        ["goal", `${(gx * 100).toFixed(1)} cm fwd`],
        ["", `${gy >= 0 ? "+" : ""}${(gy * 100).toFixed(1)} cm lat`],
        ["heading", `${headingDeg >= 0 ? "+" : ""}${headingDeg.toFixed(1)}\u00B0`],
        ["index", `${st.goal_indices[1]} of ${st.spatial.length - 1}`],
      ];
      rows.forEach(([k, v], i) => {
        const y = 16 + i * 14;
        if (k) label(ctx, k, 8, y, { align: "left", size: 10, color: TEXT_DIM });
        label(ctx, v, 152, y, { align: "right", size: 11, color: k === "heading" ? GOAL : TEXT_MID });
      });
    }
  }
}

/* ===========================================================================
   2. CAMERA OVERLAY  (unchanged semantics, sharper rendering)
   =========================================================================== */
const camOv = document.getElementById("camera-overlay");
const on = (id) => document.getElementById(id).checked;

function drawCameraOverlay(st) {
  const fit = fitCanvas(camOv); if (!fit) return;
  const { ctx, w, h } = fit;
  ctx.clearRect(0, 0, w, h);
  const px = (p) => [p[0] * w, p[1] * h];

  if (on("ov-temporal")) {
    ctx.fillStyle = TEMPORAL;
    for (const p of st.temporal_cam) {
      if (!p) continue;
      const [u, v] = px(p);
      ctx.beginPath(); ctx.arc(u, v, 3, 0, 6.2832); ctx.fill();
    }
  }

  if (on("ov-spatial")) {
    const color = SPATIAL;   /* see drawTrajectory: no longer the command hue */
    /* a dark casing under the path so it survives a bright road surface */
    for (const [width, stroke] of [[5.5, "rgba(0,0,0,0.55)"], [2.5, color]]) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      let pen = false;
      for (const p of st.spatial_cam) {
        if (!p) { pen = false; continue; }
        const [u, v] = px(p);
        pen ? ctx.lineTo(u, v) : ctx.moveTo(u, v);
        pen = true;
      }
      ctx.stroke();
    }
    ctx.fillStyle = color;
    for (const p of st.spatial_cam) {
      if (!p) continue;
      const [u, v] = px(p);
      ctx.beginPath(); ctx.arc(u, v, 3.5, 0, 6.2832); ctx.fill();
    }
  }

  if (on("ov-goal") && st.goal_indices.length === 3) {
    const apex = st.spatial_cam[st.goal_indices[1]];
    if (apex) {
      const [u, v] = px(apex);
      ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(u, v, 8, 0, 6.2832); ctx.stroke();
      ctx.strokeStyle = GOAL; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(u, v, 8, 0, 6.2832); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(u - 12, v); ctx.lineTo(u - 10, v); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(u + 10, v); ctx.lineTo(u + 12, v); ctx.stroke();
    }
  }
}

/* ===========================================================================
   3. COMMAND HEAD
   Probabilities sum to 1, so a stacked band reads the mix directly without
   line-tracing; the current values sit underneath as labelled bars.
   =========================================================================== */
const cmdCv = document.getElementById("cmd-head");
const PROBS_LEN = 160;
const probsHistory = [];

function drawCommandHead(st) {
  probsHistory.push(st.command_probs.slice());
  if (probsHistory.length > PROBS_LEN) probsHistory.shift();

  const fit = fitCanvas(cmdCv); if (!fit) return;
  const { ctx, w, h } = fit;
  ctx.clearRect(0, 0, w, h);

  const bandH = Math.round(h * 0.56);
  const barsTop = bandH + 14;

  /* --- stacked band ---------------------------------------------------- */
  const n = probsHistory.length;
  if (n >= 2) {
    /* newest sample pinned to the right edge, history growing leftward: the
       chart reads correctly from the second sample instead of looking broken
       until the ring buffer fills (53 s at 3 Hz). */
    const step = w / (PROBS_LEN - 1);
    const xAt = (i) => w - (n - 1 - i) * step;
    const acc = new Array(n).fill(0);
    for (let cmd = 0; cmd < 4; cmd++) {
      ctx.beginPath();
      for (let i = 0; i < n; i++) ctx.lineTo(xAt(i), bandH - acc[i] * bandH);
      for (let i = n - 1; i >= 0; i--) {
        const next = acc[i] + probsHistory[i][cmd];
        ctx.lineTo(xAt(i), bandH - next * bandH);
      }
      ctx.closePath();
      ctx.fillStyle = COMMAND_COLORS[cmd];
      ctx.globalAlpha = 0.82;
      ctx.fill();
      ctx.globalAlpha = 1;
      for (let i = 0; i < n; i++) acc[i] += probsHistory[i][cmd];
    }
    /* leading edge marks "now" */
    const edge = xAt(n - 1);
    ctx.strokeStyle = TEXT;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(edge, 0); ctx.lineTo(edge, bandH); ctx.stroke();
  }

  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, bandH + 0.5); ctx.lineTo(w, bandH + 0.5); ctx.stroke();
  label(ctx, "older", 2, bandH + 11, { align: "left", size: 10 });
  label(ctx, "now", w - 2, bandH + 11, { align: "right", size: 10 });

  /* --- current values, as labelled horizontal bars --------------------- */
  const rowH = (h - barsTop) / 4;
  const nameW = 56, valW = 40;
  const trackX = nameW + 4;
  const trackW = w - trackX - valW - 4;
  const argmax = st.command_probs.indexOf(Math.max(...st.command_probs));

  for (let i = 0; i < 4; i++) {
    const y = barsTop + i * rowH;
    const mid = y + rowH / 2;
    const p = st.command_probs[i];

    label(ctx, COMMAND_NAMES[i], 0, mid + 3, {
      align: "left", size: 10.5,
      color: i === argmax ? TEXT : TEXT_DIM,
      weight: i === argmax ? 600 : 400,
    });

    const barH = Math.min(8, rowH - 4);
    ctx.fillStyle = RULE_SOFT;
    roundRectPath(ctx, trackX, mid - barH / 2, trackW, barH, barH / 2);
    ctx.fill();

    ctx.fillStyle = COMMAND_COLORS[i];
    ctx.globalAlpha = i === argmax ? 1 : 0.5;
    roundRectPath(ctx, trackX, mid - barH / 2, Math.max(barH, p * trackW), barH, barH / 2);
    ctx.fill();
    ctx.globalAlpha = 1;

    label(ctx, p.toFixed(2), w, mid + 3, {
      align: "right", size: 11,
      color: i === argmax ? TEXT : TEXT_DIM,
      weight: i === argmax ? 600 : 400,
    });
  }
}

/* ===========================================================================
   4. SPEED HEAD — bins with a labelled cap line and a bin-0 emphasis.
   =========================================================================== */
const spdCv = document.getElementById("speed-head");

function drawSpeedHead(st) {
  const fit = fitCanvas(spdCv); if (!fit) return;
  const { ctx, w, h } = fit;
  ctx.clearRect(0, 0, w, h);
  const bins = st.speed_bins.map(kmhToMps);
  if (!bins.length) return;
  const cap = kmhToMps(st.target_speed_kmh);

  const top = 16, bottom = 18, plotH = h - top - bottom;
  const yMax = Math.max(...bins, cap) * 1.18 || 1;
  const toY = (v) => h - bottom - (v / yMax) * plotH;

  /* faint grid, labelled on the left. Steps sized for a sub-1 m/s axis. */
  const gstep = yMax > 2 ? 0.5 : yMax > 0.8 ? 0.2 : yMax > 0.3 ? 0.1 : 0.05;
  for (let v = 0; v <= yMax; v += gstep) {
    const y = toY(v);
    ctx.strokeStyle = RULE_SOFT;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
  }

  /* the cap the controller will actually clamp to */
  const capY = toY(cap);
  ctx.strokeStyle = TEXT_DIM;
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, capY + 0.5); ctx.lineTo(w, capY + 0.5); ctx.stroke();
  ctx.setLineDash([]);
  label(ctx, "cap " + cap.toFixed(2), w, capY - 4, { align: "right", size: 10.5, color: TEXT_MID });

  const n = bins.length;
  const gap = 5;
  const bw = (w - gap * (n - 1)) / n;

  for (let i = 0; i < n; i++) {
    const x = i * (bw + gap);
    const bh = Math.max(2, (bins[i] / yMax) * plotH);
    const y = h - bottom - bh;

    ctx.fillStyle = TEMPORAL;
    if (i !== 0) ctx.globalAlpha = 0.34;
    roundRectPath(ctx, x, y, bw, bh, 2);
    ctx.fill();
    ctx.globalAlpha = 1;

    const tx = Math.min(Math.max(x + bw / 2, 14), w - 14);
    if (i === 0 || bw > 30) {
      label(ctx, bins[i].toFixed(2), tx, y - 4,
        { size: 10.5, color: i === 0 ? TEXT : TEXT_DIM, weight: i === 0 ? 600 : 400 });
    }
    if (i === 0 || bw > 38 || i % 2 === 0) {
      label(ctx, i === 0 ? "now" : "+" + (i * st.temporal_dt).toFixed(2),
        tx, h - 5, { size: 10 });
    }
  }
}

/* ===========================================================================
   5. SPEED TRACKING
   What the model wants vs. what the controller actually sent, against the cap.
   =========================================================================== */
const traceCv = document.getElementById("speed-trace");
const TRACE_LEN = 180;
const trace = [];

function drawSpeedTrace(st) {
  /* all three in robot m/s - unscaled, the model line sits 20x above the rest */
  trace.push([modelSpeedMps(st), kmhToMps(st.commanded_speed_kmh), kmhToMps(st.target_speed_kmh)]);
  if (trace.length > TRACE_LEN) trace.shift();

  const fit = fitCanvas(traceCv); if (!fit) return;
  const { ctx, w, h } = fit;
  ctx.clearRect(0, 0, w, h);

  const top = 14, bottom = 26, plotH = h - top - bottom;
  let yMax = 0.1;   /* just enough that an all-zero trace still has an axis */
  for (const [a, b, c] of trace) yMax = Math.max(yMax, a, b, c);
  yMax *= 1.15;
  const toY = (v) => h - bottom - (v / yMax) * plotH;
  /* same right-anchoring as the command head */
  const traceStep = w / (TRACE_LEN - 1);
  const xAt = (i) => w - (trace.length - 1 - i) * traceStep;

  const gstep = yMax > 2 ? 0.5 : yMax > 0.8 ? 0.2 : yMax > 0.3 ? 0.1 : 0.05;
  for (let v = 0; v <= yMax + 1e-9; v += gstep) {
    const y = toY(v);
    ctx.strokeStyle = RULE_SOFT;
    ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
    label(ctx, v.toFixed(gstep < 0.1 ? 2 : 1), 2, y - 3, { align: "left", size: 10 });
  }

  if (trace.length >= 2) {
    /* cap */
    ctx.strokeStyle = TEXT_DIM;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    trace.forEach(([, , cap], i) => i ? ctx.lineTo(xAt(i), toY(cap)) : ctx.moveTo(xAt(i), toY(cap)));
    ctx.stroke();
    ctx.setLineDash([]);

    /* model speed — filled area, it is the reference signal */
    ctx.beginPath();
    ctx.moveTo(xAt(0), h - bottom);
    trace.forEach(([m], i) => ctx.lineTo(xAt(i), toY(m)));
    ctx.lineTo(xAt(trace.length - 1), h - bottom);
    ctx.closePath();
    ctx.fillStyle = TEMPORAL;
    ctx.globalAlpha = 0.13;
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.strokeStyle = TEMPORAL;
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    trace.forEach(([m], i) => i ? ctx.lineTo(xAt(i), toY(m)) : ctx.moveTo(xAt(i), toY(m)));
    ctx.stroke();

    /* commanded — what actually left the controller */
    ctx.strokeStyle = TEXT;
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    trace.forEach(([, c], i) => i ? ctx.lineTo(xAt(i), toY(c)) : ctx.moveTo(xAt(i), toY(c)));
    ctx.stroke();

    /* emphasised endpoints */
    const last = trace[trace.length - 1];
    const ex = xAt(trace.length - 1);
    for (const [v, col] of [[last[0], TEMPORAL], [last[1], TEXT]]) {
      ctx.fillStyle = WELL;
      ctx.beginPath(); ctx.arc(ex, toY(v), 3.5, 0, 6.2832); ctx.fill();
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(ex, toY(v), 2.25, 0, 6.2832); ctx.fill();
    }
  }

  label(ctx, "■ model", 0, h - 6, { align: "left", size: 10.5, color: TEMPORAL });
  label(ctx, "■ commanded", 64, h - 6, { align: "left", size: 10.5, color: TEXT_MID });
  label(ctx, "m/s", w, h - 6, { align: "right", size: 10 });
}

/* ===========================================================================
   LINK WATCHDOG
   A dropped WiFi link usually does NOT fire onclose: TCP keeps the half-open
   socket alive for minutes, so the page shows stale data and you steer blind.
   What we watch is the age of the last payload actually received.
   =========================================================================== */
const STATE_STALE_MS = 3000;    // clears the worst measured RTT (~1.1 s each way)
const CAMERA_STALE_MS = 4000;   // camera is throttled to 10 Hz, allow more slack
const RECONNECT_MS = 1000;

let lastStateAt = 0;
let lastCameraAt = 0;
let linkDown = null;            // null = never painted, so the first pass always runs

function renderLinkState() {
  const now = Date.now();
  const stateAge = now - lastStateAt;
  const down = lastStateAt === 0 || stateAge > STATE_STALE_MS;

  const banner = document.getElementById("link-banner");
  if (down) {
    const secs = lastStateAt === 0 ? null : (stateAge / 1000).toFixed(1);
    banner.textContent = secs === null
      ? "NO CONNECTION - waiting for data..."
      : `CONNECTION LOST - no data for ${secs}s${
          now - lastCameraAt > CAMERA_STALE_MS ? " (no camera feed either)" : ""}`;
  }
  if (down !== linkDown) {
    banner.classList.toggle("visible", down);
    document.body.classList.toggle("link-down", down);
    const pill = document.getElementById("link-pill");
    pill.dataset.state = down ? "down" : "up";
    if (down) document.getElementById("link-text").textContent = "no link";
    linkDown = down;
  }
}
setInterval(renderLinkState, 250);

/* rate of actually-received state payloads, independent of the model */
let stateFrames = 0;
let stateRateAt = 0;

function bumpLinkRate() {
  const now = Date.now();
  stateFrames++;
  if (now - stateRateAt > 1000) {
    if (!linkDown) {
      document.getElementById("link-text").textContent =
        "link " + (stateFrames * 1000 / (now - stateRateAt)).toFixed(1) + " Hz";
    }
    stateFrames = 0;
    stateRateAt = now;
  }
}

/* ===========================================================================
   SOCKETS
   =========================================================================== */
function connectCamera() {
  const sock = new WebSocket(`ws://${window.location.host}/ws/camera`);
  sock.binaryType = "blob";
  sock.onmessage = (event) => {
    lastCameraAt = Date.now();
    const img = document.getElementById("camera");
    const url = URL.createObjectURL(event.data);
    img.src = url;
    img.onload = () => URL.revokeObjectURL(url);
  };
  sock.onclose = () => setTimeout(connectCamera, RECONNECT_MS);
  sock.onerror = () => sock.close();
}
connectCamera();

function connectState() {
  const sock = new WebSocket(`ws://${window.location.host}/ws/state`);
  sock.onmessage = (event) => {
    // Stamp arrival BEFORE rendering: a payload that arrives but fails to draw
    // still proves the link is alive.
    lastStateAt = Date.now();
    bumpLinkRate();
    try {
      applyState(JSON.parse(event.data));
    } catch (err) {
      console.error("state render failed", err);
    } finally {
      // Ask for the next payload only once this one is drawn - that IS the
      // backpressure. In finally so a throwing render cannot stall the stream.
      if (sock.readyState === WebSocket.OPEN) sock.send("next");
    }
  };
  sock.onclose = () => setTimeout(connectState, RECONNECT_MS);
  sock.onerror = () => sock.close();
}
connectState();

function post(path, body) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch((err) => console.error(path, err));
}

/* ===========================================================================
   STATUS RAIL
   =========================================================================== */
const RAM_LOW_MB = 500;       // amber: getting thin
const RAM_CRITICAL_MB = 250;  // red: the next big allocation swaps

function setChip(el, cmd, name) {
  el.className = "chip c" + cmd;
  el.textContent = name;
}

function renderRam(mem) {
  const el = document.getElementById("cell-ram");
  const unit = document.getElementById("cell-ram-unit");
  const bar = document.getElementById("cell-ram-bar");
  if (!mem) {
    el.textContent = "-";
    el.className = "n";
    unit.textContent = "";
    bar.style.width = "0%";
    return;
  }
  el.textContent = String(mem.used_mb);
  unit.textContent = `MB of ${mem.total_mb}`;
  // Headroom, not swap-in-use, says the robot is in trouble: once pages are out
  // the kernel leaves them there long after the pressure is gone.
  const level = mem.available_mb < RAM_CRITICAL_MB ? " is-critical"
    : mem.available_mb < RAM_LOW_MB ? " is-warn" : "";
  el.className = "n" + level;
  el.title = `${mem.available_mb} MB available, ${mem.swap_used_mb} MB swapped out`;
  bar.style.width = ((mem.used_mb / mem.total_mb) * 100).toFixed(1) + "%";
  bar.className = level.trim();
}

function renderPaused() {
  const el = document.getElementById("cell-inference");
  el.textContent = "paused";
  el.className = "n is-paused";
  document.getElementById("cell-inference-unit").hidden = true;
  const bar = document.getElementById("cell-budget");
  bar.style.width = "0%";
  bar.className = "";
}

function renderStatus(st) {
  setChip(document.getElementById("cell-fed"), st.command, st.command_name);
  setChip(document.getElementById("cell-action"), st.action, st.action_name);
  const speedCell = document.getElementById("cell-speed");
  speedCell.textContent = modelSpeedMps(st).toFixed(2);
  speedCell.title = `${st.speed_kmh.toFixed(1)} km/h in model (CARLA) frame`;
  document.getElementById("cell-cmdspeed").textContent =
    kmhToMps(st.commanded_speed_kmh).toFixed(2);

  const el = document.getElementById("cell-inference");
  el.textContent = st.inference_ms.toFixed(1);
  el.className = "n";
  document.getElementById("cell-inference-unit").hidden = false;

  // ms spent vs. the period the requested rate allows
  const frac = Math.min(1, st.inference_ms / (1000 / st.inference_hz));
  const bar = document.getElementById("cell-budget");
  bar.style.width = (frac * 100).toFixed(1) + "%";
  bar.className = frac > 0.95 ? "is-critical" : frac > 0.75 ? "is-warn" : "";

  renderRam(st.memory);
}

/* ===========================================================================
   RUN CONTROLS
   =========================================================================== */
function setRunning(flag) {
  document.getElementById("btn-start").setAttribute("aria-pressed", String(flag));
  document.getElementById("btn-stop").setAttribute("aria-pressed", String(!flag));
  const el = document.getElementById("run-state");
  el.textContent = flag ? "running" : "stopped";
  el.dataset.on = String(flag);
}
document.getElementById("btn-start").addEventListener("click", () => post("/enabled", { enabled: true }));
document.getElementById("btn-stop").addEventListener("click", () => post("/enabled", { enabled: false }));

const KEY_TO_CMD = { a: 1, w: 2, d: 3, s: 0 };

function feedCommand(key) {
  post("/command", { key });
}
for (const btn of document.querySelectorAll(".key")) {
  btn.addEventListener("click", () => feedCommand(btn.dataset.key));
}
window.addEventListener("keydown", (event) => {
  if (event.repeat || event.metaKey || event.ctrlKey) return;
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "")) return;
  const key = event.key.toLowerCase();
  if (!(key in KEY_TO_CMD)) return;
  event.preventDefault();
  feedCommand(key);
});

/* ===========================================================================
   TUNING CONTROLS
   One table instead of ten near-identical apply/restore blocks. Readout ids are
   "out-<el>"; `from` converts the ROS field into what the slider shows.
   =========================================================================== */
const CONTROLS = [
  { el: "speed-cap",      post: "/target_speed",    key: "target_speed_kmh",
    body: (v) => ({ speed_kmh: v * KMH_PER_MPS }), from: (v) => v / KMH_PER_MPS, dp: 2 },
  { el: "steer-floor",    post: "/steer_min_speed", key: "steer_min_speed_kmh",
    body: (v) => ({ speed_kmh: v * KMH_PER_MPS }), from: (v) => v / KMH_PER_MPS, dp: 2 },
  { el: "skip-waypoints", post: "/skip_waypoints",  key: "skip_waypoints",
    body: (v) => ({ index: Math.round(v) }), dp: 0 },
  { el: "max-lookat",     post: "/max_lookat",      key: "max_lookat",
    body: (v) => ({ index: Math.round(v) }), dp: 0 },
  { el: "goal-beta",      post: "/goal_beta",       key: "goal_beta",
    body: (v) => ({ beta: v }), dp: 2 },
  { el: "inference-hz",   post: "/inference_hz",    key: "inference_hz",
    body: (v) => ({ hz: v }), dp: 1 },
  { el: "command-beta",   post: "/command_beta",    key: "command_beta",
    body: (v) => ({ beta: v }), dp: 2 },
  { el: "speed-beta",     post: "/speed_beta",      key: "speed_beta",
    body: (v) => ({ beta: v }), dp: 2 },
];

// Bumped name: the pre-redesign cookie held km/h where these sliders now hold
// m/s, and restoring it would silently set a 10x cap.
const SETTINGS_COOKIE = "duckad_controls_v2";
const seeded = new Set();

function paintControl(c) {
  const input = document.getElementById(c.el);
  document.getElementById("out-" + c.el).textContent =
    parseFloat(input.value).toFixed(c.dp);
}

function pushControl(c) {
  post(c.post, c.body(parseFloat(document.getElementById(c.el).value)));
}

for (const c of CONTROLS) {
  document.getElementById(c.el).addEventListener("input", () => {
    seeded.add(c.el);
    paintControl(c);
    pushControl(c);
    saveSettings();
  });
  paintControl(c);
}

const pidIds = ["pid-kp", "pid-ki", "pid-kd"];
function pushPid() {
  const [kp, ki, kd] = pidIds.map((id) => parseFloat(document.getElementById(id).value));
  post("/pid", { kp, ki, kd });
  saveSettings();
}
for (const id of pidIds) {
  document.getElementById(id).addEventListener("change", () => {
    seeded.add("pid");
    pushPid();
  });
}

const speedToggle = document.getElementById("use-speed-controller");
speedToggle.addEventListener("change", () => {
  seeded.add("use-speed-controller");
  post("/use_speed_controller", { enabled: speedToggle.checked });
  saveSettings();
});

function saveSettings() {
  const values = {};
  for (const c of CONTROLS) values[c.el] = document.getElementById(c.el).value;
  for (const id of pidIds) values[id] = document.getElementById(id).value;
  values["use-speed-controller"] = speedToggle.checked;
  document.cookie = `${SETTINGS_COOKIE}=${encodeURIComponent(JSON.stringify(values))}` +
    "; max-age=31536000; path=/; samesite=lax";
}

(function restoreSettings() {
  const match = document.cookie.match(new RegExp("(?:^|; )" + SETTINGS_COOKIE + "=([^;]*)"));
  if (!match) return;
  let saved;
  try {
    saved = JSON.parse(decodeURIComponent(match[1]));
  } catch (err) {
    console.error("failed to parse settings cookie", err);
    return;
  }
  for (const c of CONTROLS) {
    if (saved[c.el] === undefined) continue;
    document.getElementById(c.el).value = saved[c.el];
    paintControl(c);
    pushControl(c);
    seeded.add(c.el);
  }
  if (pidIds.every((id) => saved[id] !== undefined)) {
    for (const id of pidIds) document.getElementById(id).value = saved[id];
    pushPid();
    seeded.add("pid");
  }
  if (saved["use-speed-controller"] !== undefined) {
    speedToggle.checked = saved["use-speed-controller"];
    post("/use_speed_controller", { enabled: speedToggle.checked });
    seeded.add("use-speed-controller");
  }
})();

// Anything the cookie did not restore takes its value from the robot, once.
function seedFromState(st) {
  for (const c of CONTROLS) {
    if (seeded.has(c.el) || st[c.key] === undefined) continue;
    const input = document.getElementById(c.el);
    input.value = c.from ? c.from(st[c.key]) : st[c.key];
    paintControl(c);
    seeded.add(c.el);
  }
  if (!seeded.has("pid") && st.lateral_pid) {
    pidIds.forEach((id, i) => { document.getElementById(id).value = st.lateral_pid[i]; });
    seeded.add("pid");
  }
  if (!seeded.has("use-speed-controller") && st.use_speed_controller !== undefined) {
    speedToggle.checked = st.use_speed_controller;
    seeded.add("use-speed-controller");
  }
}

/* ===========================================================================
   WEIGHTS PICKER
   Options come from the server's listing of model_node/assets, so whatever is
   in the folder is what you can pick. `model_busy` (from model_node) is what
   drives the paused state - not a local timer.
   =========================================================================== */
const modelSelect = document.getElementById("model-select");
const swapNote = document.getElementById("swap-note");
const modelSizes = {};
let lastModelFile = "";
let pendingModel = null;
let wasBusy = false;

fetch("/models")
  .then((r) => r.json())
  .then(({ models }) => {
    for (const m of models) {
      modelSizes[m.file] = m.size_mb;
      const opt = document.createElement("option");
      opt.value = m.file;
      opt.textContent = m.file.replace(/\.onnx$/, "");
      modelSelect.appendChild(opt);
    }
    showActiveModel(lastModelFile);
  })
  .catch((err) => console.error("/models", err));

function showActiveModel(file) {
  // Only assign when it actually differs: this runs on every state payload and
  // must not fight a select the user has open.
  if (file && modelSelect.value !== file
      && [...modelSelect.options].some((o) => o.value === file)) {
    modelSelect.value = file;
  }
  const mb = modelSizes[modelSelect.value];
  document.getElementById("out-model-size").textContent =
    mb === undefined ? "\u2014" : mb.toFixed(1) + " MB";
}

modelSelect.addEventListener("change", () => {
  // Stop before switching. controller_node also stops itself on ~busy, but the
  // UI should not keep claiming "running" through a reload.
  pendingModel = modelSelect.value;
  post("/enabled", { enabled: false });
  post("/model", { file: pendingModel });
});

function renderSwapping(busy, file) {
  modelSelect.disabled = busy;
  swapNote.classList.toggle("visible", busy);
  document.body.classList.toggle("swapping", busy);
  if (busy) {
    document.getElementById("swap-text").textContent =
      "Inference paused · loading " + (pendingModel || modelSelect.value);
  }
  // Whatever the outcome, the node's file is the truth once the swap ends - so
  // a load that failed and fell back reverts the select by itself.
  if (wasBusy && !busy) pendingModel = null;
  wasBusy = busy;
  if (!busy && !pendingModel) showActiveModel(file);
}

/* ===========================================================================
   OVERLAY LAYER TOGGLES
   =========================================================================== */
const OVERLAY_KEY = "duckad-overlay-layers";
(function initOverlays() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(OVERLAY_KEY)) || {}; } catch (_) {}
  for (const id of ["ov-spatial", "ov-temporal", "ov-goal"]) {
    const box = document.getElementById(id);
    if (saved[id] !== undefined) box.checked = saved[id];
    box.addEventListener("change", () => {
      const state = {};
      for (const other of ["ov-spatial", "ov-temporal", "ov-goal"]) {
        state[other] = document.getElementById(other).checked;
      }
      try { localStorage.setItem(OVERLAY_KEY, JSON.stringify(state)); } catch (_) {}
      if (LAST_STATE) drawCameraOverlay(LAST_STATE);
    });
  }
})();

/* ===========================================================================
   RENDER
   =========================================================================== */
let LAST_STATE = null;

function applyState(st) {
  lastModelFile = st.model_file;
  renderSwapping(st.model_busy, st.model_file);
  setRunning(st.enabled);
  seedFromState(st);

  if (st.model_busy) {
    // No session, no predictions: hold the last frame. The link and host RAM
    // are still real, so they keep reporting.
    renderPaused();
    renderRam(st.memory);
    return;
  }

  LAST_STATE = st;
  renderStatus(st);
  drawCameraOverlay(st);
  drawTrajectory(st);
  drawCommandHead(st);
  drawSpeedHead(st);
  drawSpeedTrace(st);
}

// Canvases are sized from their CSS box, so a resize has to redraw them. Only
// the stateless ones: the command head and speed trace append to ring buffers,
// and they repaint on the next payload anyway.
window.addEventListener("resize", () => {
  if (!LAST_STATE) return;
  drawCameraOverlay(LAST_STATE);
  drawTrajectory(LAST_STATE);
  drawSpeedHead(LAST_STATE);
});
