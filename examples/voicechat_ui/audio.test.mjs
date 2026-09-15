// SPDX-License-Identifier: Apache-2.0
import test from "node:test";
import assert from "node:assert/strict";
import {
    PCMFramer,
    PCMPlayer,
    PlaybackResampler,
    encodePCM,
    decodePCM,
} from "./audio.mjs";

for (const rate of [16000, 44100, 48000]) {
    test(`${rate} Hz capture preserves duration across worklet boundaries`, () => {
        const framer = new PCMFramer(rate);
        const count = rate * 0.16;
        const input = new Float32Array(count).fill(0.5);
        const frames = [];
        for (let start = 0; start < count; start += 128)
            framer.push(input.subarray(start, start + 128), (f) =>
                frames.push(f),
            );
        assert.equal(frames.length, 2);
        assert.equal(framer.offset, 0);
        for (const frame of frames)
            assert.ok(frame.every((v) => Math.abs(v - 16384) <= 1));
    });
}

test("fractional resampling is independent of client block boundaries", () => {
    const input = Float32Array.from({ length: 44100 }, (_, i) =>
        Math.sin(i * 0.05),
    );
    function sample(block) {
        const framer = new PCMFramer(44100);
        const frames = [];
        for (let start = 0; start < input.length; start += block)
            framer.push(input.subarray(start, start + block), (f) =>
                frames.push(...f),
            );
        return frames;
    }
    assert.deepEqual(sample(128), sample(997));
});

test("PCM transport preserves signed little-endian values", () => {
    const values = new Int16Array([-32768, -123, 0, 123, 32767]);
    const decoded = decodePCM(encodePCM(values));
    assert.deepEqual(
        [...decoded].map((x) => Math.round(x * 32768)),
        [...values],
    );
    assert.throws(() => decodePCM(btoa("x")), /Invalid PCM16/);
});

function fakeContext() {
    const sources = [];
    return {
        sources,
        currentTime: 0,
        sampleRate: 22050,
        destination: {},
        createAnalyser: () => ({ connect() {}, disconnect() {} }),
        createBuffer: (_, n, rate) => ({
            duration: n / rate,
            copyToChannel() {},
        }),
        createBufferSource: () => {
            const source = {
                connect() {},
                disconnect() {},
                start(at) {
                    this.at = at;
                },
                stop() {
                    this.stopped = true;
                },
            };
            sources.push(source);
            return source;
        },
        async close() {},
    };
}
const event = { response_id: "r", item_id: "i", sglang: { epoch: 0 } };

test("playback schedules contiguous audio at the output sample rate", () => {
    const ctx = fakeContext();
    const acks = [];
    const player = new PCMPlayer(ctx, (_, ms) => acks.push(ms));
    player.enqueue(new Float32Array(1764), 22050, event);
    player.enqueue(new Float32Array(1764), 22050, event);
    assert.ok(Math.abs(ctx.sources[1].at - ctx.sources[0].at - 0.08) < 1e-9);
    ctx.sources.forEach((s) => s.onended());
    assert.deepEqual(acks, [80, 160]);
});

test("interrupt clears scheduled audio and never acknowledges cancelled samples", () => {
    const ctx = fakeContext();
    const acks = [];
    const player = new PCMPlayer(ctx, (_, ms) => acks.push(ms));
    player.enqueue(new Float32Array(1764), 22050, event);
    player.clear();
    assert.equal(player.queuedSeconds, 0);
    assert.ok(ctx.sources[0].stopped);
    ctx.sources[0].onended();
    assert.deepEqual(acks, []);
    player.enqueue(new Float32Array(1764), 22050, {
        ...event,
        sglang: { epoch: 1 },
    });
    ctx.sources[1].onended();
    assert.deepEqual(acks, [80]);
});

test("arrival jitter does not insert gaps into continuous PCM", () => {
    const ctx = fakeContext();
    const player = new PCMPlayer(ctx);
    for (let i = 0; i < 12; i++) {
        ctx.currentTime = i * 0.085 + (i % 2 ? 0.01 : 0);
        player.enqueue(new Float32Array(1764), 22050, event);
    }
    for (let i = 1; i < ctx.sources.length; i++) {
        assert.ok(
            Math.abs(ctx.sources[i].at - ctx.sources[i - 1].at - 0.08) < 1e-9,
        );
    }
    assert.equal(player.rebuffers, 0);
});

test("a genuine underrun replenishes jitter headroom once", () => {
    const ctx = fakeContext();
    const player = new PCMPlayer(ctx);
    player.enqueue(new Float32Array(1764), 22050, event);
    ctx.currentTime = 0.8;
    player.enqueue(new Float32Array(1764), 22050, event);
    ctx.currentTime = 0.88;
    player.enqueue(new Float32Array(1764), 22050, event);
    assert.equal(player.rebuffers, 1);
    assert.ok(Math.abs(ctx.sources[2].at - ctx.sources[1].at - 0.08) < 1e-9);
});

test("playback resampling is continuous across arbitrary packet boundaries", () => {
    const pcm = Float32Array.from(
        { length: 22050 },
        (_, i) => 0.2 * Math.sin((2 * Math.PI * 437 * i) / 22050),
    );
    for (const rate of [44100, 48000]) {
        const expected = new PlaybackResampler(22050, rate).push(pcm);
        const streaming = new PlaybackResampler(22050, rate);
        const actual = [];
        for (let i = 0; i < pcm.length; i += 137)
            actual.push(...streaming.push(pcm.subarray(i, i + 137)));
        assert.deepEqual(Float32Array.from(actual), expected);
        assert.ok(actual.every(Number.isFinite));
        assert.ok(Math.abs(actual.length / rate - pcm.length / 22050) < 0.001);
    }
});

test("fallback device rate uses native-rate buffers and clears filter state on cancel", () => {
    const ctx = fakeContext();
    ctx.sampleRate = 48000;
    const player = new PCMPlayer(ctx);
    player.enqueue(new Float32Array(1764).fill(0.1), 22050, event);
    assert.ok(player.resampler);
    player.clear();
    assert.equal(player.resampler, null);
    player.enqueue(new Float32Array(1764), 22050, {
        ...event,
        sglang: { epoch: 1 },
    });
    assert.ok(player.resampler.buffer.every((x) => x === 0));
});
