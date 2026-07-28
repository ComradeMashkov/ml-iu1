import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  MISSION_DURATION_S,
  makeMissionSeries,
  missionKinematics,
  phaseAt,
  policyMetrics,
  signalAt,
  splitMetrics,
  windowMetrics,
} from "../assets/flight-mission-core.mjs";

test("mission phases cover the full timeline", () => {
  assert.equal(phaseAt(0).id, "taxi");
  assert.equal(phaseAt(180).id, "maneuver");
  assert.equal(phaseAt(MISSION_DURATION_S).id, "landing");
});

test("kinematics reaches cruise altitude and returns to runway", () => {
  assert.ok(missionKinematics(120).altitudeM > 1900);
  assert.ok(missionKinematics(300).altitudeM < 1);
});

test("fault injection increases the late-flight degradation score", () => {
  const healthy = signalAt(210, { enabled: false });
  const degraded = signalAt(210, {
    enabled: true,
    noisePct: 20,
    biasA: 2,
    driftC: 12,
    dropoutPct: 0,
    delayS: 1.2,
  });
  assert.ok(degraded.probability > healthy.probability);
});

test("disabling injection removes every configured fault, including noise", () => {
  const disabledWithValues = signalAt(210, {
    enabled: false,
    noisePct: 60,
    biasA: 6,
    driftC: 24,
    dropoutPct: 25,
    delayS: 2.5,
  });
  const healthy = signalAt(210, { enabled: false });
  assert.deepEqual(disabledWithValues, healthy);
});

test("series is deterministic and complete", () => {
  const first = makeMissionSeries();
  const second = makeMissionSeries();
  assert.equal(first.length, 301);
  assert.deepEqual(first, second);
});

test("policy counts always cover the validation set", () => {
  const metrics = policyMetrics(0.55, 25, 2);
  assert.equal(metrics.tp + metrics.fp + metrics.fn + metrics.tn, metrics.sampleSize);
});

test("shorter stride increases overlap and window count", () => {
  const dense = windowMetrics(4, 0.5);
  const sparse = windowMetrics(4, 3);
  assert.ok(dense.overlap > sparse.overlap);
  assert.ok(dense.count > sparse.count);
});

test("group and time split remove simulated neighbour leakage", () => {
  assert.ok(splitMetrics("random", 0.75).leakage > 0.5);
  assert.equal(splitMetrics("group", 0.75).leakage, 0);
  assert.equal(splitMetrics("time", 0.75).leakage, 0);
});

test("every required DOM id exists exactly once", async () => {
  const source = await readFile(
    new URL("../assets/flight-mission.mjs", import.meta.url),
    "utf8",
  );
  const html = await readFile(
    new URL("../interactive/flight-mission.html", import.meta.url),
    "utf8",
  );
  const requiredIds = new Set(
    [...source.matchAll(/\$\("([^"]+)"\)/g)].map((match) => match[1]),
  );
  const htmlIds = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  const uniqueHtmlIds = new Set(htmlIds);

  assert.equal(htmlIds.length, uniqueHtmlIds.size, "HTML contains duplicate ids");
  requiredIds.forEach((id) => {
    assert.ok(uniqueHtmlIds.has(id), `missing DOM element #${id}`);
  });
});
