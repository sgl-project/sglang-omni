import test from "node:test";
import assert from "node:assert/strict";

import { DuplexSession, heardTextEnd } from "./session.js";

const PROGRESS = [
  { textEnd: 2, audioMs: 400 },
  { textEnd: 5, audioMs: 900 },
  { textEnd: 9, audioMs: 1500 },
];

test("the heard text is the largest text_end whose audio has played, stepwise", () => {
  assert.deepEqual(
    [0, 399, 400, 899, 900, 1499, 1500, 9000].map((playedMs) => heardTextEnd(PROGRESS, playedMs)),
    [0, 0, 2, 2, 5, 5, 9, 9],
  );
  let previous = 0;
  for (let playedMs = 0; playedMs <= 2000; playedMs += 10) {
    const end = heardTextEnd(PROGRESS, playedMs);
    assert.ok(end >= previous);
    previous = end;
  }
});

function openSession(text = "One, two, three.", progress = PROGRESS) {
  const sent = [];
  const session = new DuplexSession({ transport: (event) => sent.push(event), now: () => 0 });
  session.handle({ type: "session.created", session: { id: "s" } });
  session.handle({ type: "session.updated", session: { sglang: { granted: { input_audio_format: { rate: 16000 }, supports_truncate: true } } } });
  session.handle({ type: "response.created", response: { id: "resp_0" } });
  session.handle({ type: "response.output_audio_transcript.delta", response_id: "resp_0", delta: text });
  session.handle({ type: "response.output_audio.delta", response_id: "resp_0", item_id: "item_0", delta: "" });
  for (const point of progress) {
    session.handle({ type: "sglang.response.progress", response_id: "resp_0", item_id: "item_0", output_index: 0, content_index: 0, text_end: point.textEnd, audio_ms: point.audioMs });
  }
  return { session, sent };
}

test("an interruption freezes the heard text where the truncate cuts the reply", () => {
  const { session, sent } = openSession();
  session.played("resp_0", 600, false);
  assert.equal(session.heardEnd("resp_0"), 2);
  session.played("resp_0", 950, true);
  assert.equal(session.heardEnd("resp_0"), 5);
  assert.deepEqual(sent.at(-1), { type: "conversation.item.truncate", event_id: sent.at(-1).event_id, item_id: "item_0", content_index: 0, audio_end_ms: 950 });
  session.played("resp_0", 1600, false);
  session.handle({ type: "sglang.response.progress", response_id: "resp_0", item_id: "item_0", text_end: 12, audio_ms: 1400 });
  assert.equal(session.heardEnd("resp_0"), 5);
  assert.equal(sent.filter((event) => event.type === "conversation.item.truncate").length, 1);
});

test("a reply played to its end sends no truncate and is heard in full", () => {
  const { session, sent } = openSession();
  session.played("resp_0", 2000, false);
  assert.equal(session.heardEnd("resp_0"), 9);
  assert.ok(sent.every((event) => event.type !== "conversation.item.truncate"));
});

test("a reply that ends cut keeps only its heard prefix, nothing when none of it played", () => {
  const { session } = openSession();
  session.played("resp_0", 950, true);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, ", pending: "" });
  session.handle({ type: "response.done", response: { id: "resp_0", status: "cancelled" } });
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, ", pending: "" });

  const silent = openSession().session;
  silent.played("resp_0", 0, true);
  silent.handle({ type: "response.done", response: { id: "resp_0", status: "cancelled" } });
  assert.deepEqual(silent.captionParts("resp_0"), { heard: "", pending: "" });
});

test("a session sends only output_modalities unless instructions, turn detection or sglang settings are given", () => {
  const sent = [];
  const transport = (event) => sent.push(JSON.parse(JSON.stringify(event)));
  new DuplexSession({ transport, now: () => 0 }).handle({ type: "session.created", session: { id: "s" } });
  assert.deepEqual(sent.at(-1).session, { output_modalities: ["audio"] });
  new DuplexSession({ transport, now: () => 0, outputModalities: ["text"], instructions: "Answer briefly.", extension: { sampling: { temperature: 0.2 } }, turnDetection: { type: "server_vad", silence_duration_ms: 800 } }).handle({ type: "session.created", session: { id: "s" } });
  assert.deepEqual(sent.at(-1).session, {
    output_modalities: ["text"],
    instructions: "Answer briefly.",
    audio: { input: { turn_detection: { type: "server_vad", silence_duration_ms: 800 } } },
    sglang: { sampling: { temperature: 0.2 } },
  });
});

// The interrupted replies below come from a recorded session of the qwen3-asr-qwen3-tts-customvoice profile.
const STOCK_REPLY = "I cannot access live stock market data, but I can explain trading basics or discuss recent market trends.";
const MODEL_REPLY = "I’m Qwen, a large language model independently developed by Alibaba Group’s Tongyi Lab.";

function interrupt(session, playedMs) {
  session.played("resp_0", playedMs, true);
  session.handle({ type: "response.done", response: { id: "resp_0", status: "cancelled" } });
}

// The characters of the cut reply the server's history kept, sent again whenever the cut moves.
function heard(session, textEnd) {
  session.handle({ type: "sglang.response.heard", response_id: "resp_0", item_id: "item_0", text_end: textEnd });
}

test("a reply cut before any progress does not show its unplayed text as heard", () => {
  const { session } = openSession(STOCK_REPLY, []);
  interrupt(session, 128);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "", pending: "" });
  heard(session, 1);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "I", pending: "" });
});

test("a reply cut inside its first sentence keeps the prefix the history kept", () => {
  const { session } = openSession(MODEL_REPLY, [{ textEnd: 87, audioMs: 6560 }]);
  interrupt(session, 2352);
  heard(session, 3);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "I’m", pending: "" });
});

test("a completed reply the player stops keeps the prefix the history kept", () => {
  const { session } = openSession(MODEL_REPLY, [{ textEnd: 87, audioMs: 6560 }]);
  session.handle({ type: "response.done", response: { id: "resp_0", status: "completed" } });
  session.played("resp_0", 2352, true);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "", pending: "" });
  heard(session, 3);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "I’m", pending: "" });
});

test("text_end counts code points, so a character outside the BMP is kept whole", () => {
  const { session } = openSession("A𠮷B", [{ textEnd: 2, audioMs: 400 }]);
  session.played("resp_0", 500, false);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "A𠮷", pending: "B" });
  interrupt(session, 500);
  heard(session, 3);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "A𠮷B", pending: "" });
});

test("late updates after a cut neither erase the heard text nor restore cancelled text", () => {
  const { session } = openSession(MODEL_REPLY, [{ textEnd: 87, audioMs: 6560 }]);
  interrupt(session, 2352);
  heard(session, 3);
  session.handle({ type: "sglang.response.progress", response_id: "resp_0", item_id: "item_0", text_end: 40, audio_ms: 3000 });
  session.handle({ type: "response.output_audio_transcript.delta", response_id: "resp_0", delta: " More." });
  assert.deepEqual(session.captionParts("resp_0"), { heard: "I’m", pending: "" });
});

test("a completed reply keeps the heard part and the rest", () => {
  const { session } = openSession();
  session.handle({ type: "response.done", response: { id: "resp_0", status: "completed" } });
  session.played("resp_0", 600, false);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "On", pending: "e, two, three." });
  session.played("resp_0", 2000, false);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, two,", pending: " three." });
});
