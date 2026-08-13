const test = require("node:test");
const assert = require("node:assert/strict");

const {
  RealtimePlaybackController,
  RealtimeTurnTracker,
} = require("./playback.js");

class FakeSource {
  constructor() {
    this.stopped = false;
    this.onended = null;
  }

  connect() {}
  start() {}
  stop() {
    this.stopped = true;
  }
}

class FakeContext {
  constructor() {
    this.state = "running";
    this.currentTime = 0;
    this.destination = {};
    this.sources = [];
    this.closed = false;
  }

  createBuffer(_channels, length, sampleRate) {
    return { duration: length / sampleRate, copyToChannel() {} };
  }

  createBufferSource() {
    const source = new FakeSource();
    this.sources.push(source);
    return source;
  }

  close() {
    this.state = "closed";
    this.closed = true;
  }

  resume() {}
}

function controller() {
  const contexts = [];
  const playback = new RealtimePlaybackController({
    createAudioContext: () => {
      const context = new FakeContext();
      contexts.push(context);
      return context;
    },
    decodeBase64: () => new Uint8Array([0, 0, 1, 0]),
  });
  return { playback, contexts };
}

test("interrupt stops playback and rejects late audio", () => {
  const { playback, contexts } = controller();
  playback.beginResponse("response-a");
  assert.equal(
    playback.queueAudioDelta("audio", "response-a", "assistant-a"),
    true,
  );

  assert.deepEqual(playback.interrupt(), [
    { responseId: "response-a", itemId: "assistant-a", audioEndMs: 0 },
  ]);
  assert.equal(contexts[0].sources[0].stopped, true);
  assert.equal(contexts[0].closed, false);
  assert.equal(playback.queueAudioDelta("late", "response-a"), false);
});

test("speech can interrupt playback after response generation completed", () => {
  const { playback } = controller();
  playback.beginResponse("response-a");
  playback.queueAudioDelta("audio", "response-a", "assistant-a");
  playback.finishResponse("response-a", "completed");

  assert.deepEqual(playback.interrupt(), [
    { responseId: "response-a", itemId: "assistant-a", audioEndMs: 0 },
  ]);
});

test("a new response can play after an interrupted response", () => {
  const { playback, contexts } = controller();
  playback.beginResponse("response-a");
  playback.queueAudioDelta("audio", "response-a");
  playback.interrupt();
  playback.beginResponse("response-b");

  assert.equal(playback.queueAudioDelta("audio", "response-b"), true);
  assert.equal(contexts.length, 1);
});

test("explicit close releases the audio context", () => {
  const { playback, contexts } = controller();
  playback.beginResponse("old-response");
  playback.queueAudioDelta("audio", "old-response", "old-item");
  playback.interrupt();
  playback.beginResponse("new-response");

  playback.close();

  assert.equal(contexts[0].closed, true);
  assert.deepEqual(playback.interrupt(), []);
  assert.equal(playback.cancelledResponseIds.size, 0);
  assert.equal(playback.responseMetadata.size, 0);
});

test("interrupt reports the assistant item and played duration", () => {
  const { playback, contexts } = controller();
  playback.beginResponse("response-a");
  playback.queueAudioDelta("audio", "response-a", "assistant-a");
  contexts[0].currentTime = 0.27;
  playback.finishResponse("response-a", "completed");

  assert.deepEqual(playback.interrupt(), [
    {
      responseId: "response-a",
      itemId: "assistant-a",
      audioEndMs: 250,
    },
  ]);
});

test("finished playback interruptions do not retain rejected IDs", () => {
  const { playback } = controller();
  for (let index = 0; index < 100; index++) {
    const responseId = `response-${index}`;
    playback.beginResponse(responseId);
    playback.queueAudioDelta("audio", responseId, `assistant-${index}`);
    playback.finishResponse(responseId, "completed");
    playback.interrupt();
  }

  assert.equal(playback.cancelledResponseIds.size, 0);
});

test("pending-start interruption marks the later response stale", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-a");
  turns.markPendingInterrupted();

  const binding = turns.beginResponse("response-a");

  assert.deepEqual(binding, { itemId: "item-a", interrupted: true });
  assert.equal(turns.ownsResponse("response-a"), false);
  assert.deepEqual(turns.finishResponse("response-a"), {
    itemId: "item-a",
    interrupted: true,
  });
});

test("pending interruption is cleared when speech stops", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-a");
  turns.markPendingInterrupted();
  turns.clearPendingInterruption();

  assert.deepEqual(turns.beginResponse("response-a"), {
    itemId: "item-a",
    interrupted: false,
  });
});

test("turn-detected terminal remains interrupted after pending state clears", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-a");
  turns.markPendingInterrupted();
  turns.clearPendingInterruption();
  turns.beginResponse("response-a");

  assert.deepEqual(turns.finishResponse("response-a", "turn_detected"), {
    itemId: "item-a",
    interrupted: true,
  });
});

test("a response delayed until speech stops remains live", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-a");

  assert.deepEqual(turns.beginResponse("response-a"), {
    itemId: "item-a",
    interrupted: false,
  });
  assert.equal(turns.ownsResponse("response-a"), true);
});

test("active interruption retains response-to-turn correlation", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-a");
  turns.beginResponse("response-a");

  assert.equal(turns.interruptResponse("response-a"), "item-a");
  assert.deepEqual(turns.finishResponse("response-a"), {
    itemId: "item-a",
    interrupted: true,
  });
});

test("a stale terminal cannot finish a newer live response", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-live");
  turns.beginResponse("response-live");

  assert.deepEqual(turns.finishResponse("response-stale"), {
    itemId: null,
    interrupted: false,
  });
  assert.equal(turns.ownsResponse("response-live"), true);
});

test("interrupting finished responses does not grow stale state", () => {
  const turns = new RealtimeTurnTracker();
  for (let index = 0; index < 100; index++) {
    const itemId = `item-${index}`;
    const responseId = `response-${index}`;
    turns.commit(itemId);
    turns.beginResponse(responseId);
    turns.finishResponse(responseId);
    assert.equal(turns.interruptResponse(responseId), itemId);
  }

  assert.equal(turns.staleResponseIds.size, 0);
  assert.ok(turns.responseItems.size <= 64);
});
