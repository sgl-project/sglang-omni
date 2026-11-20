import test from "node:test";
import assert from "node:assert/strict";

import { DuplexSession, heardTextEnd, playingTextEnd } from "./session.js";

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

test("a playing caption moves word by word through the stretch being played", () => {
  const text = "One, two, three. Four five.";
  const progress = [
    { textEnd: 16, audioMs: 1600 },
    { textEnd: 27, audioMs: 2600 },
  ];
  assert.deepEqual(
    [0, 300, 500, 900, 1600, 2000, 2300, 2600].map((playedMs) => text.slice(0, playingTextEnd(text, progress, playedMs))),
    ["", "One", "One", "One, two", "One, two, three.", "One, two, three.", "One, two, three. Four", "One, two, three. Four five."],
  );
  assert.equal(playingTextEnd(text, progress.slice(0, 1), 2000), 16);
});

test("a playing caption advances through Chinese without spaces", () => {
  const text = "今天天气很好。我们走吧。";
  const end = playingTextEnd(text, [{ textEnd: 7, audioMs: 1400 }], 700);
  assert.ok(end > 0 && end < 7, `heard ${text.slice(0, end)}`);
});

test("a cut reply's caption stays where it stopped until the server reports what it kept", () => {
  const { session } = openSession("One, two, three.", [{ textEnd: 16, audioMs: 1600 }]);
  session.played("resp_0", 900, false);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, two", pending: ", three." });
  session.played("resp_0", 950, true);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, two", pending: "" });
  session.handle({ type: "sglang.response.heard", response_id: "resp_0", item_id: "item_0", text_end: 3 });
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One", pending: "" });
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
  assert.deepEqual(session.captionParts("resp_0"), { heard: "I’m Qwen, a large language", pending: "" });
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
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One", pending: ", two, three." });
  session.played("resp_0", 2000, false);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, two,", pending: " three." });
});

// The caption parts with each run of spaces left by a screen block read as one space.
function captionWords(session, responseId) {
  const { heard, pending } = session.captionParts(responseId);
  return { heard: heard.replace(/\s+/g, " ").trim(), pending: pending.replace(/\s+/g, " ").trim() };
}

test("a screen block stays out of the caption, heard or still to play, and the server's offsets still fit", () => {
  const text = "On screen. <screen>\nbrew install python\n</screen>\nThen run it.";
  const blockEnd = text.indexOf("Then");
  const { session } = openSession(text, [
    { textEnd: 10, audioMs: 500 },
    { textEnd: blockEnd, audioMs: 500 },
    { textEnd: text.length, audioMs: 1500 },
  ]);
  session.played("resp_0", 500, false);
  assert.deepEqual(captionWords(session, "resp_0"), { heard: "On screen.", pending: "Then run it." });
  session.played("resp_0", 1500, false);
  assert.deepEqual(captionWords(session, "resp_0"), { heard: "On screen. Then run it.", pending: "" });
});

test("the caption waits for a tag still arriving instead of showing it", () => {
  const { session } = openSession("Look here <scr", []);
  assert.deepEqual(captionWords(session, "resp_0"), { heard: "Look here", pending: "" });
});

test("the screen shows a reply's blocks once a progress point past them has played, so it never runs ahead of the audio", () => {
  const { session } = openSession("Here. <screen>a</screen>", [{ textEnd: 24, audioMs: 400 }]);
  session.played("resp_0", 400, false);
  assert.equal(session.screen, "a");
  // The text arrives before the first progress event, and the caption shows it whole until then.
  const text = "Run this. <screen>lsof -i :8080</screen> Then check.";
  session.handle({ type: "response.created", response: { id: "resp_1" } });
  session.handle({ type: "response.output_audio_transcript.delta", response_id: "resp_1", delta: text });
  session.handle({ type: "response.output_audio.delta", response_id: "resp_1", item_id: "item_1", delta: "" });
  assert.equal(session.screen, "a");
  // The server sums audio_ms chunk by chunk, so the last point can sit a rounding error past the player's count.
  for (const [textEnd, audioMs] of [[9, 500], [text.length, 1500.0000000000002]]) {
    session.handle({ type: "sglang.response.progress", response_id: "resp_1", item_id: "item_1", output_index: 0, content_index: 0, text_end: textEnd, audio_ms: audioMs });
  }
  // Late in "Then check." the caption has passed the block, but no progress point past it has played.
  session.played("resp_1", 1400, false);
  assert.ok(session.captionParts("resp_1").heard.includes("Then"));
  assert.equal(session.screen, "a");
  session.played("resp_1", 1500, false);
  assert.equal(session.screen, "lsof -i :8080");
});

test("after a cut the screen shows the blocks in the prefix the server's history kept", () => {
  const text = "Run this. <screen>lsof -i :8080</screen> Then check.";
  const { session } = openSession(text, [{ textEnd: 9, audioMs: 500 }, { textEnd: text.length, audioMs: 1500 }]);
  session.played("resp_0", 1500, true);
  assert.equal(session.screen, "lsof -i :8080");
  session.handle({ type: "sglang.response.heard", response_id: "resp_0", item_id: "item_0", output_index: 0, content_index: 0, text_end: 9 });
  assert.equal(session.screen, "");
});

function openGranted(interaction) {
  const sent = [];
  const session = new DuplexSession({ transport: (event) => sent.push(event), now: () => 0 });
  session.handle({ type: "session.created", session: { id: "s" } });
  session.handle({ type: "session.updated", session: { sglang: { granted: { input_audio_format: { rate: 16000 }, interaction } } } });
  return { session, cancels: () => sent.filter((event) => event.type === "response.cancel").length };
}

test("a local interrupt cancels the open reply of a turn-based server only", () => {
  const { session, cancels } = openGranted("turn_based");
  session.cancelReply();
  assert.equal(cancels(), 0);
  session.handle({ type: "response.created", response: { id: "resp_0" } });
  session.cancelReply();
  assert.equal(cancels(), 1);
  session.handle({ type: "response.done", response: { id: "resp_0", status: "cancelled" } });
  session.cancelReply();
  assert.equal(cancels(), 1);

  const native = openGranted("native");
  native.session.handle({ type: "response.created", response: { id: "resp_0" } });
  native.session.cancelReply();
  assert.equal(native.cancels(), 0);
});
