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

function openSession() {
  const sent = [];
  const session = new DuplexSession({ transport: (event) => sent.push(event), now: () => 0 });
  session.handle({ type: "session.created", session: { id: "s" } });
  session.handle({ type: "session.updated", session: { sglang: { granted: { input_audio_format: { rate: 16000 }, supports_truncate: true } } } });
  session.handle({ type: "response.created", response: { id: "resp_0" } });
  session.handle({ type: "response.output_audio_transcript.delta", response_id: "resp_0", delta: "One, two, three." });
  session.handle({ type: "response.output_audio.delta", response_id: "resp_0", item_id: "item_0", delta: "" });
  for (const point of PROGRESS) {
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
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, ", pending: "two, three." });
  session.handle({ type: "response.done", response: { id: "resp_0", status: "cancelled" } });
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, ", pending: "" });

  const silent = openSession().session;
  silent.played("resp_0", 0, true);
  silent.handle({ type: "response.done", response: { id: "resp_0", status: "cancelled" } });
  assert.deepEqual(silent.captionParts("resp_0"), { heard: "", pending: "" });
});

test("a completed reply keeps the heard part and the rest", () => {
  const { session } = openSession();
  session.handle({ type: "response.done", response: { id: "resp_0", status: "completed" } });
  session.played("resp_0", 600, false);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "On", pending: "e, two, three." });
  session.played("resp_0", 2000, false);
  assert.deepEqual(session.captionParts("resp_0"), { heard: "One, two,", pending: " three." });
});
