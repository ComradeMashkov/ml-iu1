export const MISSION_DURATION_S = 300;

export const MISSION_PHASES = Object.freeze([
  { id: "taxi", label: "Руление", start: 0, end: 25 },
  { id: "takeoff", label: "Взлёт", start: 25, end: 52 },
  { id: "climb", label: "Набор высоты", start: 52, end: 94 },
  { id: "cruise", label: "Маршрут", start: 94, end: 154 },
  { id: "maneuver", label: "Манёвр", start: 154, end: 205 },
  { id: "descent", label: "Снижение", start: 205, end: 272 },
  { id: "landing", label: "Посадка", start: 272, end: 300 },
]);

export const FLIGHT_WAYPOINTS = Object.freeze([
  { t: 0, x: -126, y: 1.4, z: -7 },
  { t: 25, x: -82, y: 1.4, z: -7 },
  { t: 52, x: -13, y: 35, z: -2 },
  { t: 94, x: 76, y: 79, z: 1 },
  { t: 124, x: 145, y: 86, z: 11 },
  { t: 154, x: 212, y: 81, z: 34 },
  { t: 180, x: 222, y: 73, z: 91 },
  { t: 205, x: 170, y: 65, z: 135 },
  { t: 238, x: 70, y: 40, z: 105 },
  { t: 272, x: -14, y: 8, z: 39 },
  { t: 300, x: -72, y: 1.4, z: 1 },
]);

export const DEFAULT_FAULTS = Object.freeze({
  enabled: true,
  noisePct: 12,
  biasA: 1.2,
  driftC: 8,
  dropoutPct: 2,
  delayS: 0.6,
});

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const smoothstep = (edge0, edge1, value) => {
  const x = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return x * x * (3 - 2 * x);
};

const noiseAt = (t, channel) => {
  const x = Math.sin((t + 17 * channel) * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
};

export function phaseAt(timeS) {
  const t = clamp(timeS, 0, MISSION_DURATION_S);
  return MISSION_PHASES.find((phase) => t >= phase.start && t < phase.end)
    ?? MISSION_PHASES.at(-1);
}

export function missionKinematics(timeS) {
  const t = clamp(timeS, 0, MISSION_DURATION_S);
  let altitudeM;
  let speedKmh;
  let bankDeg = 0;
  let pitchDeg = 0;

  if (t < 25) {
    altitudeM = 0;
    speedKmh = 18 + 1.35 * t;
  } else if (t < 52) {
    const p = (t - 25) / 27;
    altitudeM = 340 * p * p;
    speedKmh = 52 + 172 * p;
    pitchDeg = 12 * Math.sin(Math.PI * p);
  } else if (t < 94) {
    const p = (t - 52) / 42;
    altitudeM = 340 + 1660 * p;
    speedKmh = 224 + 42 * p;
    pitchDeg = 8 * (1 - p);
  } else if (t < 154) {
    const p = (t - 94) / 60;
    altitudeM = 2000 + 80 * Math.sin(Math.PI * p);
    speedKmh = 266 + 8 * Math.sin(2 * Math.PI * p);
  } else if (t < 205) {
    const p = (t - 154) / 51;
    altitudeM = 2000 - 230 * p + 45 * Math.sin(2 * Math.PI * p);
    speedKmh = 272 + 24 * Math.sin(Math.PI * p);
    bankDeg = 38 * Math.sin(Math.PI * p);
    pitchDeg = -3 + 5 * Math.sin(2 * Math.PI * p);
  } else if (t < 272) {
    const p = (t - 205) / 67;
    altitudeM = 1770 * (1 - p) + 90 * p;
    speedKmh = 250 * (1 - p) + 125 * p;
    pitchDeg = -7 * Math.sin(Math.PI * p);
    bankDeg = -12 * Math.sin(2 * Math.PI * p);
  } else {
    const p = (t - 272) / 28;
    altitudeM = 90 * (1 - smoothstep(0, 0.78, p));
    speedKmh = 125 * (1 - p) + 28 * p;
    pitchDeg = -4 * Math.sin(Math.PI * p);
  }

  return {
    altitudeM: Math.max(0, altitudeM),
    speedKmh,
    bankDeg,
    pitchDeg,
    phase: phaseAt(t),
  };
}

function actuatorCommand(timeS) {
  const t = clamp(timeS, 0, MISSION_DURATION_S);
  let command = 1.5 * Math.sin(t / 9) + 0.7 * Math.sin(t / 3.8);
  if (t >= 25 && t < 52) command += 3.6 * Math.sin((t - 25) / 4.4);
  if (t >= 154 && t < 205) {
    command += 8.5 * Math.sin((t - 154) / 3.7)
      + 2.4 * Math.sin((t - 154) / 1.25);
  }
  if (t >= 272) command -= 3.2 * Math.sin((t - 272) / 2.8);
  return command;
}

export function signalAt(timeS, faults = DEFAULT_FAULTS) {
  const t = clamp(timeS, 0, MISSION_DURATION_S);
  const enabled = Boolean(faults.enabled);
  const faultProgress = enabled ? smoothstep(164, 218, t) : 0;
  const noiseScale = enabled ? (Number(faults.noisePct) || 0) / 100 : 0;
  const delayS = enabled ? Number(faults.delayS) || 0 : 0;
  const commandDeg = actuatorCommand(t);
  const delayedCommand = actuatorCommand(Math.max(0, t - 0.45 - delayS * faultProgress));
  const dynamicLag = 0.42 * Math.sin(t / 5.7) + 0.18 * Math.sin(t / 1.9);
  const degradation = faultProgress * (1.4 + 0.9 * Math.sin(t / 7.5));
  const positionDeg = delayedCommand - dynamicLag - degradation
    + noiseScale * 1.1 * noiseAt(t, 1);
  const trackingErrorDeg = commandDeg - positionDeg;
  const maneuverLoad = phaseAt(t).id === "maneuver" ? 4.6 : 0;
  const currentA = 7.2
    + 0.58 * Math.abs(commandDeg)
    + 0.92 * Math.abs(trackingErrorDeg)
    + maneuverLoad
    + faultProgress * (Number(faults.biasA) || 0)
    + noiseScale * 2.2 * noiseAt(t, 2);
  const temperatureC = 36
    + 0.047 * t
    + 0.34 * currentA
    + faultProgress * (Number(faults.driftC) || 0)
    + noiseScale * 1.6 * noiseAt(t, 3);
  const turbulence = 1.2 * Math.exp(-((t - 188) ** 2) / 230);
  const vibrationG = Math.max(
    0.03,
    0.16
      + 0.012 * Math.abs(commandDeg)
      + turbulence
      + 0.65 * faultProgress
      + noiseScale * 0.42 * noiseAt(t, 4),
  );
  const logit = -5.0
    + 0.55 * Math.abs(trackingErrorDeg)
    + 0.16 * Math.max(0, currentA - 11)
    + 0.10 * Math.max(0, temperatureC - 53)
    + 1.65 * Math.max(0, vibrationG - 0.25);
  const probability = 1 / (1 + Math.exp(-logit));
  const dropoutGate = noiseAt(Math.floor(t * 4), 9);
  const isDropout = enabled
    && t >= 164
    && dropoutGate > 1 - 2 * ((Number(faults.dropoutPct) || 0) / 100);

  return {
    timeS: t,
    commandDeg,
    positionDeg: isDropout ? Number.NaN : positionDeg,
    trackingErrorDeg: isDropout ? Number.NaN : trackingErrorDeg,
    currentA: isDropout ? Number.NaN : currentA,
    temperatureC: isDropout ? Number.NaN : temperatureC,
    vibrationG: isDropout ? Number.NaN : vibrationG,
    probability: isDropout ? Number.NaN : probability,
    isDropout,
  };
}

export function makeMissionSeries(faults = DEFAULT_FAULTS, stepS = 1) {
  const series = [];
  for (let timeS = 0; timeS <= MISSION_DURATION_S + 1e-9; timeS += stepS) {
    series.push(signalAt(timeS, faults));
  }
  return series;
}

const VALIDATION_LABELS = Object.freeze([
  0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
  0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
]);

const VALIDATION_SCORES = Object.freeze([
  0.04, 0.08, 0.12, 0.18, 0.23, 0.28, 0.31, 0.36,
  0.39, 0.45, 0.48, 0.53, 0.56, 0.59, 0.62, 0.66,
  0.69, 0.72, 0.75, 0.78, 0.82, 0.86, 0.91, 0.95,
]);

export function policyMetrics(threshold = 0.5, costFn = 25, costFp = 2) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;

  VALIDATION_LABELS.forEach((label, index) => {
    const prediction = VALIDATION_SCORES[index] >= threshold ? 1 : 0;
    if (label === 1 && prediction === 1) tp += 1;
    if (label === 0 && prediction === 1) fp += 1;
    if (label === 1 && prediction === 0) fn += 1;
    if (label === 0 && prediction === 0) tn += 1;
  });

  return {
    tp,
    fp,
    fn,
    tn,
    recall: tp + fn ? tp / (tp + fn) : 0,
    precision: tp + fp ? tp / (tp + fp) : 0,
    cost: costFn * fn + costFp * fp,
    sampleSize: VALIDATION_LABELS.length,
  };
}

export function windowMetrics(lengthS = 2, strideS = 0.5) {
  const length = clamp(Number(lengthS) || 2, 0.5, 12);
  const stride = clamp(Number(strideS) || 0.5, 0.1, length);
  const overlap = clamp(1 - stride / length, 0, 0.99);
  const count = Math.max(
    0,
    1 + Math.floor((MISSION_DURATION_S - length) / stride),
  );
  return {
    lengthS: length,
    strideS: stride,
    overlap,
    count,
    sharedSeconds: Math.max(0, length - stride),
    heuristicEffectiveCount: Math.max(1, Math.round(count * (1 - overlap * 0.86))),
  };
}

export function splitMetrics(mode = "random", overlap = 0.75) {
  const safeOverlap = clamp(overlap, 0, 0.99);
  const scenarios = {
    random: {
      label: "Случайные окна",
      neighbourOverlapIndex: clamp(0.42 + 0.58 * safeOverlap, 0, 0.99),
      illustrativeF1: 0.83 + 0.13 * safeOverlap,
      question: "Узнаём соседнее окно того же полёта?",
    },
    group: {
      label: "Новый полёт",
      neighbourOverlapIndex: 0,
      illustrativeF1: 0.78,
      question: "Обобщаемся на новый flight_id?",
    },
    time: {
      label: "Будущее",
      neighbourOverlapIndex: 0,
      illustrativeF1: 0.72,
      question: "Работаем после временного сдвига?",
    },
  };
  return scenarios[mode] ?? scenarios.random;
}
