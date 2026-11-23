import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const workletSource = readFileSync(new URL("./worklet.js", import.meta.url), "utf8");

function createPlayback(sampleRate) {
  const statuses = [];
  const processors = new Map();
  vm.runInNewContext(workletSource, {
    sampleRate,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = { postMessage: (status) => statuses.push(status) };
      }
    },
    registerProcessor: (name, processor) => processors.set(name, processor),
  });
  const processor = new (processors.get("playback-processor"))({});
  return {
    statuses,
    send: (message) => processor.port.onmessage({ data: message }),
    render: (sampleCount = 128) => {
      const output = new Float32Array(sampleCount);
      for (let offset = 0; offset < sampleCount; offset += 128) {
        processor.process([], [[output.subarray(offset, offset + 128)]]);
      }
      return output;
    },
  };
}

function replayCadence(sampleRate, intervalSeconds = 1) {
  const playback = createPlayback(sampleRate);
  playback.send({ type: "config", nativeUnitSamples: sampleRate });
  playback.send({ type: "open" });
  const gaps = [];
  let nextChunk = 0;
  let sounded = false;
  let gapSamples = 0;
  let audibleSamples = 0;
  for (let offset = 0; offset < sampleRate * 12; offset += 128) {
    if (offset >= nextChunk) {
      playback.send({ type: "push", samples: new Float32Array(sampleRate).fill(0.2), responseId: "reply" });
      nextChunk += sampleRate * intervalSeconds;
    }
    for (const sample of playback.render()) {
      if (Math.abs(sample) > 0.001) {
        audibleSamples += 1;
        sounded = true;
        if (gapSamples) gaps.push(gapSamples);
        gapSamples = 0;
      } else if (sounded) gapSamples += 1;
    }
  }
  return { gaps, audibleSamples, statuses: playback.statuses };
}

for (const sampleRate of [24000, 48000]) {
  test(`one-second native chunks stay continuous at ${sampleRate} Hz`, () => {
    const { gaps, audibleSamples, statuses } = replayCadence(sampleRate);
    const gapMilliseconds = gaps.map((samples) => Math.round(samples / sampleRate * 1000));
    assert.ok(gaps.every((samples) => samples <= 128), `interior gaps in milliseconds: ${JSON.stringify(gapMilliseconds)}`);
    assert.ok(audibleSamples >= sampleRate * 12 - 128 * 12, `only ${audibleSamples / sampleRate} audible seconds from 12 seconds of PCM`);
    assert.ok(statuses.every((status) => status.rate === 1), "nominal chunks must not trigger catch-up");
    assert.equal(statuses.at(-1).droppedMs, 0);
  });
}

test("a native unit does not increase the 300 ms startup buffer", () => {
  const playback = createPlayback(24000);
  playback.send({ type: "config", nativeUnitSamples: 24000 });
  playback.send({ type: "push", samples: new Float32Array(4800).fill(0.2) });
  assert.ok(playback.render().every((sample) => sample === 0));
  playback.send({ type: "push", samples: new Float32Array(2400).fill(0.2) });
  assert.ok(playback.render().every((sample) => sample > 0.1));
  assert.equal(playback.statuses[0].targetMs, 300);
});

test("accumulated backlog still accelerates, skips and obeys the hard cap", () => {
  const playback = createPlayback(24000);
  playback.send({ type: "config", nativeUnitSamples: 24000 });
  playback.send({ type: "push", samples: new Float32Array(48000).fill(0.2) });
  playback.render();
  assert.equal(playback.statuses.at(-1).rate, 1.25);
  assert.equal(playback.statuses.at(-1).droppedMs, 0);

  const skipping = createPlayback(24000);
  skipping.send({ type: "config", nativeUnitSamples: 24000, maxSamples: 96000 });
  skipping.send({ type: "push", samples: new Float32Array(72000).fill(0.2) });
  skipping.render();
  assert.equal(skipping.statuses.at(-1).droppedMs, 1300);
  assert.ok(skipping.statuses.at(-1).queueMs > 1600);

  const capped = createPlayback(24000);
  capped.send({ type: "config", nativeUnitSamples: 240000 });
  capped.send({ type: "push", samples: new Float32Array(72000).fill(0.2) });
  capped.render();
  assert.equal(capped.statuses.at(-1).droppedMs, 1500);
  assert.ok(capped.statuses.at(-1).queueMs <= 1500);
});

test("genuinely late chunks still report underruns", () => {
  const { gaps, statuses } = replayCadence(24000, 1.2);
  assert.ok(gaps.some((samples) => samples >= 24000 * 0.15));
  assert.ok(statuses.at(-1).underruns >= 5);
  assert.equal(statuses.at(-1).responseOpen, true);
});

test("unknown or invalid native units retain the default catch-up policy", () => {
  for (const nativeUnitSamples of [undefined, 0, -1, NaN, Infinity, 1.5, "24000"]) {
    const playback = createPlayback(24000);
    playback.send({ type: "config", nativeUnitSamples });
    playback.send({ type: "push", samples: new Float32Array(24000).fill(0.2) });
    playback.render();
    assert.equal(playback.statuses.at(-1).rate, 1.12);
  }
});

test("flush drains a short response without counting terminal silence as an underrun", () => {
  const playback = createPlayback(24000);
  playback.send({ type: "open" });
  playback.send({ type: "push", samples: new Float32Array(2400).fill(0.2) });
  playback.send({ type: "flush" });
  const output = playback.render(4800);
  assert.equal(output.filter((sample) => sample > 0.1).length, 2400);
  assert.equal(playback.statuses.at(-1).underruns, 0);
  assert.equal(playback.statuses.at(-1).responseOpen, false);
});

test("cancel removes only its response and clear stops all queued audio", () => {
  const playback = createPlayback(24000);
  playback.send({ type: "config", nativeUnitSamples: 24000 });
  playback.send({ type: "push", samples: new Float32Array(12000).fill(0.2), responseId: "old" });
  playback.send({ type: "push", samples: new Float32Array(12000).fill(-0.2), responseId: "new" });
  playback.render();
  playback.send({ type: "cancel", responseId: "old" });
  const output = playback.render(12000);
  assert.ok(output.filter((sample) => sample > 0).length <= 240);
  assert.ok(output.filter((sample) => sample < -0.1).length >= 11760);
  playback.send({ type: "clear" });
  assert.ok(playback.render(480).subarray(240).every((sample) => sample === 0));
});

test("reset starts a fresh generation with fresh counters and status energy", () => {
  const playback = createPlayback(24000);
  playback.send({ type: "open" });
  playback.send({ type: "push", samples: new Float32Array(72000).fill(0.2) });
  playback.render();
  assert.ok(playback.statuses.at(-1).droppedMs > 0);
  playback.render();
  const previousCount = playback.statuses.length;
  playback.send({ type: "reset", generation: 7 });
  const output = playback.render();
  assert.equal(playback.statuses.length, previousCount + 1);
  const status = playback.statuses.at(-1);
  assert.equal(status.generation, 7);
  assert.equal(status.responseOpen, false);
  for (const counter of ["droppedMs", "underruns", "playedMs", "queueMs"]) assert.equal(status[counter], 0);
  assert.equal(status.rate, 1);
  assert.equal(status.buffering, true);
  const energy = output.reduce((total, sample) => total + sample * sample, 0);
  assert.ok(Math.abs(status.outputRms - Math.sqrt(energy / output.length)) < 1e-8);
});

for (const sampleRate of [24000, 48000]) {
  test(`turn-based cold, bursty and long replies keep every sample at ${sampleRate} Hz`, () => {
    const playback = createPlayback(sampleRate);
    playback.send({ type: "config", turnBased: true, nativeUnitSamples: sampleRate / 5 });
    playback.send({ type: "open" });
    assert.ok(playback.render(sampleRate).every((sample) => sample === 0));

    const speech = Float32Array.from({ length: sampleRate * 12 }, (_, index) => (index % 1000 + 1) / 1024);
    playback.send({ type: "push", samples: speech.slice(0, sampleRate / 10), responseId: "reply" });
    assert.ok(playback.render().every((sample) => sample === 0));
    playback.send({ type: "push", samples: speech.slice(sampleRate / 10, sampleRate), responseId: "reply" });
    const rendered = new Float32Array(speech.length);
    rendered.set(playback.render(sampleRate));
    // The next burst arrives after playback has run dry.
    assert.ok(playback.render(sampleRate).every((sample) => sample === 0));
    for (let offset = sampleRate; offset < speech.length; offset += sampleRate * 2) {
      playback.send({ type: "push", samples: speech.slice(offset, offset + sampleRate * 2), responseId: "reply" });
    }
    playback.send({ type: "trim", keepSamples: sampleRate / 2 });
    playback.send({ type: "flush" });
    for (let offset = sampleRate; offset < rendered.length; offset += 128) {
      rendered.set(playback.render(Math.min(128, rendered.length - offset)), offset);
    }
    assert.equal(rendered.findIndex((sample, index) => sample !== speech[index]), -1, "playback must preserve the complete sample sequence");
    assert.ok(playback.statuses.every((status) => status.rate === 1 && status.droppedMs === 0));
    assert.ok(playback.render().every((sample) => sample === 0));
  });
}

test("turn-based cancellation preserves the next reply and clear stops playback", () => {
  const playback = createPlayback(24000);
  playback.send({ type: "config", turnBased: true });
  playback.send({ type: "push", samples: new Float32Array(72000).fill(0.2), responseId: "old" });
  playback.send({ type: "push", samples: new Float32Array(48000).fill(-0.2), responseId: "new" });
  playback.render();
  playback.send({ type: "cancel", responseId: "old" });
  const output = playback.render(48240);
  assert.ok(output.subarray(240).every((sample) => sample === Math.fround(-0.2)));
  assert.equal(playback.statuses.at(-1).droppedMs, 0);
  playback.send({ type: "push", samples: new Float32Array(72000).fill(0.2), responseId: "new" });
  playback.send({ type: "clear" });
  assert.ok(playback.render(480).subarray(240).every((sample) => sample === 0));
});
