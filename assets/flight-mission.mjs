import {
  DEFAULT_FAULTS,
  FLIGHT_WAYPOINTS,
  MISSION_DURATION_S,
  MISSION_PHASES,
  makeMissionSeries,
  missionKinematics,
  phaseAt,
  policyMetrics,
  signalAt,
  splitMetrics,
  windowMetrics,
} from "./flight-mission-core.mjs";

const root = document.getElementById("flight-mission-lab");
const $ = (id) => document.getElementById(id);
const query = new URLSearchParams(window.location.search);
const initialView = query.get("view") || "mission";
const isEmbed = query.get("embed") === "1";
const allowedViews = new Set(["flight", "mission", "decision", "window", "split", "fault"]);

const state = {
  view: allowedViews.has(initialView) ? initialView : "mission",
  timeS: 0,
  playing: false,
  playbackSpeed: 1,
  cameraMode: "chase",
  faults: { ...DEFAULT_FAULTS },
  series: [],
  threshold: 0.5,
  costFn: 25,
  costFp: 2,
  windowLengthS: 2,
  windowStrideS: 0.5,
  splitMode: "random",
  selectedSignals: new Set([
    "position",
    "current",
    "temperature",
    "vibration",
    "probability",
  ]),
};

document.body.classList.toggle("is-embed", isEmbed);
document.body.dataset.view = state.view;

function formatTime(timeS) {
  const whole = Math.max(0, Math.round(timeS));
  const minutes = Math.floor(whole / 60);
  const seconds = whole % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function formatFinite(value, digits = 1, fallback = "нет данных") {
  return Number.isFinite(value) ? value.toFixed(digits) : fallback;
}

function applyView(view) {
  state.view = allowedViews.has(view) ? view : "mission";
  document.body.dataset.view = state.view;

  document.querySelectorAll("[data-views]").forEach((element) => {
    const views = element.dataset.views.split(/\s+/);
    element.hidden = !views.includes(state.view);
  });

  document.querySelectorAll("[data-view-target]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.viewTarget === state.view));
  });

  drawTelemetry();
  drawSplit();
}

function buildPhaseButtons() {
  const container = $("phase-buttons");
  MISSION_PHASES.forEach((phase) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = phase.label;
    button.dataset.phase = phase.id;
    button.addEventListener("click", () => {
      setMissionTime(phase.start + 0.2);
      if (!state.playing) updateAll();
    });
    container.append(button);
  });
}

function setMissionTime(timeS) {
  state.timeS = Math.max(0, Math.min(MISSION_DURATION_S, Number(timeS) || 0));
  $("time-range").value = String(state.timeS);
}

function setPlaying(nextPlaying) {
  state.playing = Boolean(nextPlaying);
  const button = $("play-button");
  button.setAttribute("aria-pressed", String(state.playing));
  button.textContent = state.playing ? "Пауза" : state.timeS >= MISSION_DURATION_S ? "Сначала" : "Запустить";
  if (state.playing && state.timeS >= MISSION_DURATION_S) {
    setMissionTime(0);
  }
}

function readFaultControls() {
  state.faults = {
    enabled: $("fault-enabled").checked,
    noisePct: Number($("noise-range").value),
    biasA: Number($("bias-range").value),
    driftC: Number($("drift-range").value),
    dropoutPct: Number($("dropout-range").value),
    delayS: Number($("delay-range").value),
  };
  state.series = makeMissionSeries(state.faults);

  $("noise-output").value = `${state.faults.noisePct}%`;
  $("bias-output").value = `${state.faults.biasA.toFixed(1)} A`;
  $("drift-output").value = `${state.faults.driftC.toFixed(0)} °C`;
  $("dropout-output").value = `${state.faults.dropoutPct}%`;
  $("delay-output").value = `${state.faults.delayS.toFixed(1)} с`;
}

function updateMissionReadout() {
  const kinematics = missionKinematics(state.timeS);
  const phase = kinematics.phase;
  const signal = signalAt(state.timeS, state.faults);

  $("header-phase").textContent = phase.label;
  $("header-time").textContent = formatTime(state.timeS);
  $("readout-phase").textContent = phase.label;
  $("readout-altitude").textContent = Math.round(kinematics.altitudeM).toLocaleString("ru-RU");
  $("readout-speed").textContent = Math.round(kinematics.speedKmh).toString();
  $("readout-bank").textContent = Math.round(kinematics.bankDeg).toString();
  $("time-output").value = formatTime(state.timeS);

  document.querySelectorAll("[data-phase]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.phase === phase.id));
  });

  const decision = Number.isFinite(signal.probability)
    ? signal.probability >= state.threshold
    : null;
  const decisionState = $("decision-state");
  decisionState.textContent = decision === null
    ? "Нет данных"
    : decision
      ? "Снять после посадки"
      : "Наблюдать";
  decisionState.className = `state-label${decision === null ? " warning" : decision ? " danger" : ""}`;

  $("quick-fault-score").textContent = formatFinite(signal.probability, 2);
  $("quick-fault-error").textContent = `${formatFinite(Math.abs(signal.trackingErrorDeg), 1)}°`;
  $("telemetry-accessible").textContent = [
    `Время ${formatTime(state.timeS)}.`,
    `Этап ${phase.label}.`,
    `Высота ${Math.round(kinematics.altitudeM)} метров.`,
    `Вероятность отказа ${formatFinite(signal.probability, 2)}.`,
    `Решение: ${decisionState.textContent}.`,
  ].join(" ");
}

function updatePolicy() {
  state.threshold = Number($("threshold-range").value);
  state.costFn = Number($("cost-fn-range").value);
  state.costFp = Number($("cost-fp-range").value);
  const metrics = policyMetrics(state.threshold, state.costFn, state.costFp);

  $("threshold-output").value = state.threshold.toFixed(2);
  $("cost-fn-output").value = String(state.costFn);
  $("cost-fp-output").value = String(state.costFp);
  $("policy-cost").textContent = `стоимость ${metrics.cost}`;
  $("policy-recall").textContent = metrics.recall.toFixed(2);
  $("policy-precision").textContent = metrics.precision.toFixed(2);
  $("quick-policy-cost").textContent = String(metrics.cost);
  $("quick-policy-recall").textContent = metrics.recall.toFixed(2);
  $("quick-policy-precision").textContent = metrics.precision.toFixed(2);

  const cells = [
    ["", "ŷ = 0", "ŷ = 1", "y = 0", metrics.tn, metrics.fp, "y = 1", metrics.fn, metrics.tp],
    ["matrix-axis", "matrix-axis", "matrix-axis", "matrix-axis", "matrix-good", "matrix-bad", "matrix-axis", "matrix-bad", "matrix-good"],
  ];
  $("confusion-matrix").replaceChildren(
    ...cells[0].map((value, index) => {
      const cell = document.createElement("span");
      cell.className = cells[1][index];
      cell.textContent = String(value);
      return cell;
    }),
  );
}

function updateWindowing() {
  state.windowLengthS = Number($("window-length-range").value);
  const strideInput = $("window-stride-range");
  strideInput.max = String(state.windowLengthS);
  if (Number(strideInput.value) > state.windowLengthS) {
    strideInput.value = String(state.windowLengthS);
  }
  state.windowStrideS = Number(strideInput.value);
  const metrics = windowMetrics(state.windowLengthS, state.windowStrideS);
  const overlapPct = Math.round(metrics.overlap * 100);

  $("window-length-output").value = `${metrics.lengthS.toFixed(1)} с`;
  $("window-stride-output").value = `${metrics.strideS.toFixed(2).replace(/0$/, "")} с`;
  $("overlap-output").textContent = `${overlapPct}% overlap`;
  $("window-count").textContent = `${metrics.count} окон`;
  $("window-overlap").textContent = `${overlapPct}%`;
  $("window-shared").textContent = `${metrics.sharedSeconds.toFixed(2).replace(/0$/, "")} с`;
  $("window-effective").textContent = String(metrics.effectiveIndependent);
  $("quick-window-count").textContent = String(metrics.count);
  $("quick-window-effective").textContent = String(metrics.effectiveIndependent);
}

function updateSplit() {
  state.splitMode = document.querySelector('input[name="split-mode"]:checked')?.value || "random";
  const overlap = windowMetrics(state.windowLengthS, state.windowStrideS).overlap;
  const metrics = splitMetrics(state.splitMode, overlap);
  const leakagePct = Math.round(metrics.leakage * 100);

  $("split-leakage-output").textContent = metrics.leakage > 0
    ? `утечка ${leakagePct}%`
    : "группы не пересекаются";
  $("split-leakage-output").className = `state-label${metrics.leakage > 0 ? " warning" : ""}`;
  $("split-title").textContent = metrics.label;
  $("split-f1").textContent = `учебный F1 ${metrics.apparentF1.toFixed(2)}`;
  $("split-question").textContent = metrics.question;
  $("quick-split-f1").textContent = metrics.apparentF1.toFixed(2);
  $("quick-split-question").textContent = metrics.question;
}

function updateAll() {
  updateMissionReadout();
  updatePolicy();
  updateWindowing();
  updateSplit();
  updateAircraft();
  drawTelemetry();
  drawSplit();
}

function bindControls() {
  document.querySelectorAll("[data-view-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextView = button.dataset.viewTarget;
      const nextUrl = new URL(window.location.href);
      nextUrl.searchParams.set("view", nextView);
      window.history.replaceState({}, "", nextUrl);
      applyView(nextView);
      updateAll();
    });
  });

  $("play-button").addEventListener("click", () => {
    setPlaying(!state.playing);
  });
  $("time-range").addEventListener("input", (event) => {
    setMissionTime(event.target.value);
    updateAll();
  });
  $("speed-select").addEventListener("change", (event) => {
    state.playbackSpeed = Number(event.target.value);
  });

  ["threshold-range", "cost-fn-range", "cost-fp-range"].forEach((id) => {
    $(id).addEventListener("input", () => {
      updatePolicy();
      updateMissionReadout();
      drawTelemetry();
    });
  });

  ["window-length-range", "window-stride-range"].forEach((id) => {
    $(id).addEventListener("input", () => {
      updateWindowing();
      updateSplit();
      drawTelemetry();
      drawSplit();
    });
  });

  document.querySelectorAll('input[name="split-mode"]').forEach((input) => {
    input.addEventListener("change", () => {
      updateSplit();
      drawSplit();
    });
  });

  ["fault-enabled", "noise-range", "bias-range", "drift-range", "dropout-range", "delay-range"]
    .forEach((id) => {
      $(id).addEventListener("input", () => {
        readFaultControls();
        updateMissionReadout();
        drawTelemetry();
      });
    });

  document.querySelectorAll("[data-signal]").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.selectedSignals.add(input.dataset.signal);
      else state.selectedSignals.delete(input.dataset.signal);
      drawTelemetry();
    });
  });

  document.querySelectorAll("[data-camera]").forEach((button) => {
    button.addEventListener("click", () => {
      state.cameraMode = button.dataset.camera;
      document.querySelectorAll("[data-camera]").forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      updateAircraft(true);
    });
  });

  const telemetryCanvas = $("telemetry-canvas");
  telemetryCanvas.addEventListener("pointerdown", (event) => {
    const rect = telemetryCanvas.getBoundingClientRect();
    const left = Math.min(72, rect.width * 0.12);
    const right = 18;
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left - left) / (rect.width - left - right)));
    setMissionTime(ratio * MISSION_DURATION_S);
    updateAll();
  });
  telemetryCanvas.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    setMissionTime(state.timeS + (event.key === "ArrowRight" ? 1 : -1));
    updateAll();
  });
}

const plotDefinitions = [
  {
    id: "position",
    label: "привод, °",
    key: "positionDeg",
    secondaryKey: "commandDeg",
    domain: [-15, 15],
    color: "#215caf",
    secondaryColor: "#20232a",
  },
  {
    id: "current",
    label: "ток, A",
    key: "currentA",
    domain: [4, 28],
    color: "#b46a08",
  },
  {
    id: "temperature",
    label: "темп., °C",
    key: "temperatureC",
    domain: [36, 82],
    color: "#b52b34",
  },
  {
    id: "vibration",
    label: "вибр., g",
    key: "vibrationG",
    domain: [0, 2.2],
    color: "#247b75",
  },
  {
    id: "probability",
    label: "score",
    key: "probability",
    domain: [0, 1],
    color: "#7257a5",
  },
];

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function interpolatedWaypoint(timeS) {
  const time = Math.max(0, Math.min(MISSION_DURATION_S, timeS));
  let rightIndex = FLIGHT_WAYPOINTS.findIndex((point) => point.t >= time);
  if (rightIndex <= 0) return { ...FLIGHT_WAYPOINTS[0] };
  if (rightIndex < 0) return { ...FLIGHT_WAYPOINTS.at(-1) };
  const leftPoint = FLIGHT_WAYPOINTS[rightIndex - 1];
  const rightPoint = FLIGHT_WAYPOINTS[rightIndex];
  const ratio = (time - leftPoint.t) / (rightPoint.t - leftPoint.t);
  return {
    t: time,
    x: leftPoint.x + (rightPoint.x - leftPoint.x) * ratio,
    y: leftPoint.y + (rightPoint.y - leftPoint.y) * ratio,
    z: leftPoint.z + (rightPoint.z - leftPoint.z) * ratio,
  };
}

function drawFallbackAircraft(context, x, y, scale, bankDeg, headingRad = 0) {
  context.save();
  context.translate(x, y);
  context.rotate(headingRad + bankDeg * Math.PI / 180);
  context.scale(scale, scale);

  context.fillStyle = "rgba(32, 35, 42, 0.20)";
  context.beginPath();
  context.ellipse(8, 9, 43, 9, 0, 0, Math.PI * 2);
  context.fill();

  const wingGradient = context.createLinearGradient(-6, -28, 7, 28);
  wingGradient.addColorStop(0, "#17447f");
  wingGradient.addColorStop(0.48, "#2d70c8");
  wingGradient.addColorStop(1, "#17447f");
  context.fillStyle = wingGradient;
  context.beginPath();
  context.moveTo(9, -5);
  context.lineTo(-13, -34);
  context.lineTo(-20, -33);
  context.lineTo(-8, -4);
  context.lineTo(-8, 4);
  context.lineTo(-20, 33);
  context.lineTo(-13, 34);
  context.lineTo(9, 5);
  context.closePath();
  context.fill();

  const bodyGradient = context.createLinearGradient(-29, 0, 31, 0);
  bodyGradient.addColorStop(0, "#aeb9c4");
  bodyGradient.addColorStop(0.48, "#f7fafc");
  bodyGradient.addColorStop(1, "#215caf");
  context.fillStyle = bodyGradient;
  context.beginPath();
  context.moveTo(34, 0);
  context.bezierCurveTo(23, -7, -17, -7, -31, -3);
  context.lineTo(-38, 0);
  context.lineTo(-31, 3);
  context.bezierCurveTo(-17, 7, 23, 7, 34, 0);
  context.fill();

  context.fillStyle = "#20232a";
  context.beginPath();
  context.ellipse(13, 0, 8, 4, 0, 0, Math.PI * 2);
  context.fill();

  context.fillStyle = "#215caf";
  context.beginPath();
  context.moveTo(-27, -3);
  context.lineTo(-37, -13);
  context.lineTo(-40, -12);
  context.lineTo(-34, -1);
  context.lineTo(-34, 1);
  context.lineTo(-40, 12);
  context.lineTo(-37, 13);
  context.lineTo(-27, 3);
  context.closePath();
  context.fill();

  context.fillStyle = "#b52b34";
  context.fillRect(-11, 25, 8, 5);
  context.restore();
}

function drawFallbackTopView(context, width, height, currentPoint, kinematics) {
  const padding = 45;
  const xs = FLIGHT_WAYPOINTS.map((point) => point.x);
  const zs = FLIGHT_WAYPOINTS.map((point) => point.z);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  const mapX = (x) => padding + ((x - minX) / (maxX - minX)) * (width - 2 * padding);
  const mapY = (z) => height - padding - ((z - minZ) / (maxZ - minZ)) * (height - 2 * padding);

  context.fillStyle = "#879b7b";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(255, 255, 255, 0.22)";
  context.lineWidth = 1;
  for (let x = 0; x < width; x += 46) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let y = 0; y < height; y += 46) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  context.strokeStyle = "#ffffff";
  context.lineWidth = 14;
  context.beginPath();
  context.moveTo(mapX(-160), mapY(-3));
  context.lineTo(mapX(5), mapY(-3));
  context.stroke();
  context.strokeStyle = "#43484d";
  context.lineWidth = 10;
  context.stroke();

  context.strokeStyle = "#215caf";
  context.lineWidth = 3;
  context.beginPath();
  FLIGHT_WAYPOINTS.forEach((point, index) => {
    const x = mapX(point.x);
    const y = mapY(point.z);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();

  MISSION_PHASES.forEach((phase) => {
    const point = interpolatedWaypoint(phase.start);
    context.fillStyle = phase.id === kinematics.phase.id ? "#b52b34" : "#ffffff";
    context.beginPath();
    context.arc(mapX(point.x), mapY(point.z), phase.id === kinematics.phase.id ? 5 : 3, 0, Math.PI * 2);
    context.fill();
  });

  const future = interpolatedWaypoint(Math.min(MISSION_DURATION_S, state.timeS + 1));
  const heading = Math.atan2(mapY(future.z) - mapY(currentPoint.z), mapX(future.x) - mapX(currentPoint.x));
  drawFallbackAircraft(
    context,
    mapX(currentPoint.x),
    mapY(currentPoint.z),
    Math.max(0.55, Math.min(0.85, width / 900)),
    0,
    heading,
  );
}

function drawFallbackSideView(context, width, height, kinematics) {
  const horizon = height * 0.72;
  const sky = context.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, "#c9e1f4");
  sky.addColorStop(1, "#f5fafc");
  context.fillStyle = sky;
  context.fillRect(0, 0, width, horizon);
  context.fillStyle = "#879b7b";
  context.fillRect(0, horizon, width, height - horizon);

  context.strokeStyle = "rgba(33, 92, 175, 0.45)";
  context.lineWidth = 3;
  context.beginPath();
  for (let sampleTime = 0; sampleTime <= MISSION_DURATION_S; sampleTime += 2) {
    const sampleKinematics = missionKinematics(sampleTime);
    const x = 30 + (sampleTime / MISSION_DURATION_S) * (width - 60);
    const y = horizon - (sampleKinematics.altitudeM / 2200) * (horizon - 45);
    if (sampleTime === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();

  const x = 30 + (state.timeS / MISSION_DURATION_S) * (width - 60);
  const y = horizon - (kinematics.altitudeM / 2200) * (horizon - 45);
  drawFallbackAircraft(
    context,
    x,
    y,
    Math.max(0.62, Math.min(0.92, width / 820)),
    kinematics.pitchDeg * 0.45,
    0,
  );
}

function drawFallbackChaseView(context, width, height, kinematics) {
  const bankRad = -kinematics.bankDeg * Math.PI / 180 * 0.55;
  context.save();
  context.translate(width / 2, height * 0.52);
  context.rotate(bankRad);
  context.translate(-width / 2, -height * 0.52);

  const horizon = height * (0.54 + Math.max(-0.08, Math.min(0.08, kinematics.pitchDeg / 120)));
  const sky = context.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, "#b9d8ef");
  sky.addColorStop(1, "#eff7fb");
  context.fillStyle = sky;
  context.fillRect(-width, -height, width * 3, horizon + height);

  context.fillStyle = "#6f8073";
  context.beginPath();
  context.moveTo(-width, horizon + 45);
  context.lineTo(width * 0.04, horizon - 18);
  context.lineTo(width * 0.22, horizon + 8);
  context.lineTo(width * 0.43, horizon - 38);
  context.lineTo(width * 0.67, horizon + 6);
  context.lineTo(width * 0.86, horizon - 25);
  context.lineTo(width * 2, horizon + 44);
  context.lineTo(width * 2, height * 2);
  context.lineTo(-width, height * 2);
  context.closePath();
  context.fill();

  const ground = context.createLinearGradient(0, horizon, 0, height);
  ground.addColorStop(0, "#9aaa89");
  ground.addColorStop(1, "#65745f");
  context.fillStyle = ground;
  context.fillRect(-width, horizon + 25, width * 3, height * 2);

  const gridOffset = (state.timeS * 9) % 42;
  context.strokeStyle = "rgba(255, 255, 255, 0.20)";
  context.lineWidth = 1;
  for (let index = -8; index <= 8; index += 1) {
    context.beginPath();
    context.moveTo(width / 2, horizon + 18);
    context.lineTo(width / 2 + index * width * 0.18, height * 1.15);
    context.stroke();
  }
  for (let y = horizon + 32 + gridOffset; y < height * 1.12; y += 42) {
    context.beginPath();
    context.moveTo(-width, y);
    context.lineTo(width * 2, y);
    context.stroke();
  }

  if (kinematics.altitudeM < 520 || kinematics.phase.id === "landing") {
    context.fillStyle = "#43484d";
    context.beginPath();
    context.moveTo(width * 0.46, horizon + 22);
    context.lineTo(width * 0.54, horizon + 22);
    context.lineTo(width * 0.78, height * 1.06);
    context.lineTo(width * 0.22, height * 1.06);
    context.closePath();
    context.fill();
    context.strokeStyle = "#f6f3de";
    context.lineWidth = 3;
    context.setLineDash([12, 16]);
    context.beginPath();
    context.moveTo(width * 0.5, horizon + 28);
    context.lineTo(width * 0.5, height);
    context.stroke();
    context.setLineDash([]);
  }

  context.restore();

  const aircraftScale = Math.max(1.05, Math.min(1.55, width / 620));
  drawFallbackAircraft(
    context,
    width * 0.53,
    height * 0.48,
    aircraftScale,
    kinematics.bankDeg * 0.6,
    0,
  );
}

function drawFallbackScene() {
  const canvas = $("flight-backdrop");
  if (!canvas || canvas.clientWidth < 2 || canvas.clientHeight < 2) return;
  const { context, width, height } = prepareCanvas(canvas);
  context.clearRect(0, 0, width, height);
  const kinematics = missionKinematics(state.timeS);
  const currentPoint = interpolatedWaypoint(state.timeS);

  if (state.cameraMode === "top") {
    drawFallbackTopView(context, width, height, currentPoint, kinematics);
  } else if (state.cameraMode === "side") {
    drawFallbackSideView(context, width, height, kinematics);
  } else {
    drawFallbackChaseView(context, width, height, kinematics);
  }
}

function drawPath(context, definition, xScale, yScale) {
  context.beginPath();
  let drawing = false;
  state.series.forEach((sample) => {
    const value = sample[definition.key];
    if (!Number.isFinite(value)) {
      drawing = false;
      return;
    }
    const x = xScale(sample.timeS);
    const y = yScale(value);
    if (!drawing) context.moveTo(x, y);
    else context.lineTo(x, y);
    drawing = true;
  });
  context.strokeStyle = definition.color;
  context.lineWidth = 1.65;
  context.stroke();

  if (!definition.secondaryKey) return;
  context.beginPath();
  state.series.forEach((sample, index) => {
    const x = xScale(sample.timeS);
    const y = yScale(sample[definition.secondaryKey]);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = definition.secondaryColor;
  context.lineWidth = 1.1;
  context.setLineDash([5, 4]);
  context.stroke();
  context.setLineDash([]);
}

function drawTelemetry() {
  const canvas = $("telemetry-canvas");
  if (!canvas || canvas.clientWidth < 2 || canvas.clientHeight < 2) return;
  const { context, width, height } = prepareCanvas(canvas);
  const definitions = plotDefinitions.filter((definition) => state.selectedSignals.has(definition.id));
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  if (!definitions.length) {
    context.fillStyle = "#68717a";
    context.font = "13px Fira Sans";
    context.textAlign = "center";
    context.fillText("Выберите хотя бы один сигнал", width / 2, height / 2);
    return;
  }

  const left = Math.min(72, width * 0.12);
  const right = 18;
  const top = 20;
  const bottom = 25;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const laneHeight = plotHeight / definitions.length;
  const xScale = (timeS) => left + (timeS / MISSION_DURATION_S) * plotWidth;

  MISSION_PHASES.forEach((phase, index) => {
    const x0 = xScale(phase.start);
    const x1 = xScale(phase.end);
    context.fillStyle = index % 2 === 0 ? "#f5f7f9" : "#eef3f8";
    context.fillRect(x0, top, x1 - x0, plotHeight);
    if (width > 650) {
      context.fillStyle = "#68717a";
      context.font = "10px Fira Sans";
      context.textAlign = "center";
      context.fillText(phase.label, (x0 + x1) / 2, 12);
    }
  });

  definitions.forEach((definition, laneIndex) => {
    const laneTop = top + laneHeight * laneIndex;
    const laneBottom = laneTop + laneHeight;
    const [domainMin, domainMax] = definition.domain;
    const yScale = (value) => laneBottom - 7
      - ((value - domainMin) / (domainMax - domainMin)) * (laneHeight - 14);

    context.strokeStyle = "#d5dbe2";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(left, laneBottom);
    context.lineTo(width - right, laneBottom);
    context.stroke();

    context.fillStyle = "#20232a";
    context.font = "11px Fira Sans";
    context.textAlign = "right";
    context.fillText(definition.label, left - 7, laneTop + 14);

    drawPath(context, definition, xScale, yScale);

    if (definition.id === "probability") {
      context.strokeStyle = "#b52b34";
      context.lineWidth = 1;
      context.setLineDash([3, 3]);
      context.beginPath();
      context.moveTo(left, yScale(state.threshold));
      context.lineTo(width - right, yScale(state.threshold));
      context.stroke();
      context.setLineDash([]);
    }
  });

  if (state.view === "window" || state.view === "mission") {
    const metrics = windowMetrics(state.windowLengthS, state.windowStrideS);
    const currentStart = Math.floor(state.timeS / metrics.strideS) * metrics.strideS;
    for (let index = 3; index >= 0; index -= 1) {
      const start = currentStart - index * metrics.strideS;
      if (start < 0) continue;
      const x0 = xScale(start);
      const x1 = xScale(Math.min(MISSION_DURATION_S, start + metrics.lengthS));
      context.fillStyle = index === 0
        ? "rgba(33, 92, 175, 0.20)"
        : "rgba(180, 106, 8, 0.10)";
      context.fillRect(x0, top, Math.max(1, x1 - x0), 8 + index * 2);
    }
  }

  for (let tick = 0; tick <= MISSION_DURATION_S; tick += 60) {
    const x = xScale(tick);
    context.fillStyle = "#68717a";
    context.font = "10px Fira Mono";
    context.textAlign = "center";
    context.fillText(formatTime(tick), x, height - 6);
  }

  const cursorX = xScale(state.timeS);
  context.strokeStyle = "#20232a";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(cursorX, top);
  context.lineTo(cursorX, height - bottom);
  context.stroke();
  context.fillStyle = "#20232a";
  context.beginPath();
  context.arc(cursorX, top, 3.2, 0, Math.PI * 2);
  context.fill();
}

function drawSplit() {
  const canvas = $("split-canvas");
  if (!canvas || canvas.hidden || canvas.clientWidth < 2 || canvas.clientHeight < 2) return;
  const { context, width, height } = prepareCanvas(canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  const flights = ["F01", "F02", "F03", "F04", "F05", "F06"];
  const windowsPerFlight = 20;
  const left = 38;
  const right = 12;
  const top = 9;
  const rowHeight = (height - top - 9) / flights.length;
  const cellWidth = (width - left - right) / windowsPerFlight;
  const colors = { train: "#215caf", validation: "#b46a08", test: "#b52b34", gap: "#d5dbe2" };

  const roleFor = (row, column) => {
    if (state.splitMode === "random") {
      const code = (row * 17 + column * 7 + 3) % 13;
      if (code < 2) return "test";
      if (code === 2) return "validation";
      return "train";
    }
    if (state.splitMode === "group") {
      if (row === 5) return "test";
      if (row === 4) return "validation";
      return "train";
    }
    if (column < 12) return "train";
    if (column === 12 || column === 15) return "gap";
    if (column < 15) return "validation";
    return "test";
  };

  flights.forEach((flight, row) => {
    context.fillStyle = "#68717a";
    context.font = "10px Fira Mono";
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(flight, left - 6, top + rowHeight * (row + 0.5));
    for (let column = 0; column < windowsPerFlight; column += 1) {
      const role = roleFor(row, column);
      context.fillStyle = colors[role];
      context.fillRect(
        left + column * cellWidth + 0.7,
        top + row * rowHeight + 1.2,
        Math.max(1, cellWidth - 1.4),
        Math.max(2, rowHeight - 2.4),
      );
    }
  });

  const legend = [
    ["train", "train"],
    ["validation", "validation"],
    ["test", "test"],
  ];
  context.font = "10px Fira Sans";
  context.textAlign = "left";
  context.textBaseline = "alphabetic";
  let legendX = left;
  legend.forEach(([role, label]) => {
    context.fillStyle = colors[role];
    context.fillRect(legendX, height - 7, 10, 3);
    context.fillStyle = "#68717a";
    context.fillText(label, legendX + 14, height - 3);
    legendX += label.length * 6 + 35;
  });
}

let THREE;
let renderer;
let scene;
let camera;
let orbitControls;
let aircraft;
let flightCurve;
let resizeObserver;

function createAircraftModel() {
  const group = new THREE.Group();
  const blue = new THREE.MeshStandardMaterial({ color: 0x215caf, roughness: 0.56, metalness: 0.18 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x20232a, roughness: 0.48, metalness: 0.25 });
  const light = new THREE.MeshStandardMaterial({ color: 0xe9eef4, roughness: 0.62, metalness: 0.12 });
  const red = new THREE.MeshStandardMaterial({ color: 0xb52b34, roughness: 0.65 });

  const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.72, 0.9, 7.8, 20), light);
  fuselage.rotation.z = -Math.PI / 2;
  group.add(fuselage);

  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.72, 2.2, 20), blue);
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 4.95;
  group.add(nose);

  const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.72, 18, 10, 0, Math.PI * 2, 0, Math.PI / 2), dark);
  canopy.scale.set(1.45, 0.7, 0.78);
  canopy.position.set(1.25, 0.62, 0);
  group.add(canopy);

  const wing = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.18, 10.8), blue);
  wing.position.x = -0.15;
  group.add(wing);

  const tailWing = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.14, 4.1), dark);
  tailWing.position.x = -3.15;
  group.add(tailWing);

  const tail = new THREE.Mesh(new THREE.BoxGeometry(1.65, 2.2, 0.18), blue);
  tail.position.set(-3.25, 0.95, 0);
  tail.rotation.z = -0.18;
  group.add(tail);

  const actuator = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.24, 1.05), red);
  actuator.position.set(-0.8, -0.13, -4.1);
  group.add(actuator);

  group.scale.setScalar(0.9);
  return group;
}

function addEnvironment() {
  const hemisphere = new THREE.HemisphereLight(0xeaf5ff, 0x536753, 2.2);
  scene.add(hemisphere);
  const sun = new THREE.DirectionalLight(0xffffff, 2.4);
  sun.position.set(-90, 150, 80);
  scene.add(sun);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(620, 440),
    new THREE.MeshStandardMaterial({ color: 0x879b7b, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = 0;
  scene.add(ground);

  const grid = new THREE.GridHelper(520, 52, 0x5f7358, 0x91a288);
  grid.position.y = 0.035;
  scene.add(grid);

  const runway = new THREE.Mesh(
    new THREE.PlaneGeometry(180, 13),
    new THREE.MeshStandardMaterial({ color: 0x43484d, roughness: 0.98 }),
  );
  runway.rotation.x = -Math.PI / 2;
  runway.position.set(-79, 0.07, -2.8);
  scene.add(runway);

  const stripeMaterial = new THREE.MeshBasicMaterial({ color: 0xf6f3de });
  for (let index = 0; index < 17; index += 1) {
    const stripe = new THREE.Mesh(new THREE.PlaneGeometry(5, 0.45), stripeMaterial);
    stripe.rotation.x = -Math.PI / 2;
    stripe.position.set(-160 + index * 10, 0.09, -2.8);
    scene.add(stripe);
  }

  const mountainMaterial = new THREE.MeshStandardMaterial({ color: 0x718075, roughness: 1 });
  [
    [-20, 18, -120, 33],
    [55, 23, -142, 42],
    [138, 17, -115, 29],
    [205, 26, -92, 47],
    [245, 20, 5, 34],
  ].forEach(([x, radius, z, height]) => {
    const mountain = new THREE.Mesh(new THREE.ConeGeometry(radius, height, 8), mountainMaterial);
    mountain.position.set(x, height / 2 - 0.2, z);
    scene.add(mountain);
  });
}

function addFlightPath() {
  const points = FLIGHT_WAYPOINTS.map((point) => new THREE.Vector3(point.x, point.y, point.z));
  flightCurve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.35);
  const geometry = new THREE.BufferGeometry().setFromPoints(flightCurve.getPoints(420));
  const material = new THREE.LineBasicMaterial({ color: 0x215caf, transparent: true, opacity: 0.62 });
  scene.add(new THREE.Line(geometry, material));

  MISSION_PHASES.forEach((phase) => {
    const point = flightCurve.getPointAt(phase.start / MISSION_DURATION_S);
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(1.05, 12, 8),
      new THREE.MeshBasicMaterial({ color: phase.id === "maneuver" ? 0xb52b34 : 0xffffff }),
    );
    marker.position.copy(point);
    scene.add(marker);
  });
}

async function initializeThree() {
  try {
    THREE = await import("three");
    const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
    const canvas = $("flight-canvas");
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = false;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0xdbeaf7, 0);

    scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0xdbeaf7, 210, 520);
    camera = new THREE.PerspectiveCamera(46, 1, 0.1, 1100);
    camera.position.set(-144, 18, 25);

    addEnvironment();
    addFlightPath();
    aircraft = createAircraftModel();
    scene.add(aircraft);

    orbitControls = new OrbitControls(camera, canvas);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.075;
    orbitControls.enablePan = false;
    orbitControls.minDistance = 10;
    orbitControls.maxDistance = 180;
    orbitControls.maxPolarAngle = Math.PI * 0.49;
    orbitControls.enabled = false;

    resizeObserver = new ResizeObserver(() => resizeThree());
    resizeObserver.observe(canvas.parentElement);
    resizeThree();
    updateAircraft(true);
  } catch (error) {
    $("webgl-fallback").hidden = false;
    console.warn("Three.js initialization failed; 2D laboratory remains available.", error);
  }
}

function resizeThree() {
  if (!renderer || !camera) return;
  const canvas = $("flight-canvas");
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}

function updateAircraft(forceCamera = false) {
  if (!THREE || !flightCurve || !aircraft || !camera) return;
  const u = Math.max(0, Math.min(1, state.timeS / MISSION_DURATION_S));
  const position = flightCurve.getPointAt(u);
  position.y = Math.max(1.4, position.y);
  const tangent = flightCurve.getTangentAt(Math.min(0.9999, u)).normalize();
  const forward = new THREE.Vector3(1, 0, 0);
  const orientation = new THREE.Quaternion().setFromUnitVectors(forward, tangent);
  const kinematics = missionKinematics(state.timeS);

  aircraft.position.copy(position);
  aircraft.quaternion.copy(orientation);
  aircraft.rotateX(-THREE.MathUtils.degToRad(kinematics.bankDeg));

  if (orbitControls) {
    orbitControls.enabled = state.cameraMode === "orbit";
  }

  const desiredPosition = new THREE.Vector3();
  if (state.cameraMode === "chase") {
    desiredPosition.set(-20, 8, 13).applyQuaternion(aircraft.quaternion).add(position);
  } else if (state.cameraMode === "top") {
    desiredPosition.copy(position).add(new THREE.Vector3(0, 105, 0.1));
  } else if (state.cameraMode === "side") {
    desiredPosition.copy(position).add(new THREE.Vector3(-3, 12, 48));
  }

  if (state.cameraMode !== "orbit") {
    camera.position.lerp(desiredPosition, forceCamera ? 1 : 0.12);
    camera.up.set(0, 1, 0);
    camera.lookAt(position);
  } else if (orbitControls) {
    orbitControls.target.lerp(position, forceCamera ? 1 : 0.18);
  }
}

let previousFrame = performance.now();
let lastUiUpdate = 0;

function animate(now) {
  const deltaS = Math.min(0.1, Math.max(0, (now - previousFrame) / 1000));
  previousFrame = now;

  if (state.playing) {
    setMissionTime(state.timeS + deltaS * state.playbackSpeed * 5);
    if (state.timeS >= MISSION_DURATION_S) {
      setPlaying(false);
    }
  }

  updateAircraft();
  drawFallbackScene();
  if (orbitControls?.enabled) orbitControls.update(deltaS);
  if (renderer && scene && camera) renderer.render(scene, camera);

  if (state.playing && now - lastUiUpdate > 65) {
    updateMissionReadout();
    drawTelemetry();
    lastUiUpdate = now;
  }
  requestAnimationFrame(animate);
}

function initialize() {
  buildPhaseButtons();
  bindControls();
  readFaultControls();
  applyView(state.view);
  updateAll();
  initializeThree();

  const redraw = () => {
    drawFallbackScene();
    drawTelemetry();
    drawSplit();
    resizeThree();
  };
  window.addEventListener("resize", redraw);
  document.addEventListener("visibilitychange", () => {
    previousFrame = performance.now();
  });

  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches && state.view === "flight") {
    setMissionTime(92);
    updateMissionReadout();
  }
  drawFallbackScene();
  requestAnimationFrame(animate);
}

initialize();
