import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FrameCapture, frameReport } from '../src/sdf-showcase/performance.ts';

test('browser frame metrics use elapsed time, retain long frames and separate simulation speed', () => {
  const sample = frameReport([10, 10, 10, 70], .05);
  assert.equal(sample.fps, 40);
  assert.equal(sample.durationMs, 100);
  assert.equal(sample.medianMs, 10);
  assert.equal(sample.p95Ms, 70);
  assert.equal(sample.over25Ms, 1);
  assert.equal(sample.simulationRate, .5, 'fast rendering cannot conceal slow simulation');
  assert.deepEqual(sample.intervalsMs, [10, 10, 10, 70], 'raw intervals are not clipped or sorted in place');
});

test('browser capture excludes warmup and stops at its finite deadline', () => {
  const capture = new FrameCapture(500);
  assert.equal(capture.record(1400, 9), null);
  assert.equal(capture.record(1500, 9), null, 'first warm frame is the interval baseline');
  for (let t = 1600; t < 5500; t += 100) assert.equal(capture.record(t, .05), null);
  const sample = capture.record(5500, .05)!;
  assert.equal(sample.durationMs, 4000);
  assert.equal(sample.intervalsMs.length, 40);
  assert.ok(Math.abs(sample.simulationRate - .5) < 1e-12);
  assert.throws(() => capture.record(5600, .05), /finished/);
});

test('browser measurements reject missing, nonfinite or nonmonotonic samples', () => {
  for (const intervals of [[], [0], [-1], [NaN], [Infinity]]) assert.throws(() => frameReport(intervals, 0), /Invalid/);
  assert.throws(() => frameReport([16], -1), /Invalid/);
  assert.throws(() => new FrameCapture(NaN), /Invalid/);
  const capture = new FrameCapture(0);
  capture.record(1000, 0);
  assert.throws(() => capture.record(1000, 0), /increase/);
  assert.throws(() => capture.record(1200, NaN), /Invalid/);
});
