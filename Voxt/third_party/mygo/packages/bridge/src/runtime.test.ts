import { describe, expect, test } from "bun:test";
import { CallError, createRuntime } from "./runtime";
import type { Outgoing } from "./types";

type CallMsg = Extract<Outgoing, { t: "call" }>;

function setup() {
  const sent: Outgoing[] = [];
  const rt = createRuntime(
    { platform: "darwin", windowId: 7, version: "0.0.0-test", secret: "s", locale: "en", dir: "ltr" },
    (m) => sent.push(JSON.parse(m)),
  );
  return { sent, ...rt };
}

describe("call", () => {
  test("resolves with the reply for the current page", async () => {
    const { sent, runtime, internal } = setup();
    const p = runtime.call("Greeter.Greet", "bun", 1);
    const msg = sent[0] as CallMsg;
    expect(msg).toMatchObject({ t: "call", id: 1, m: "Greeter.Greet", a: ["bun", 1] });

    // A reply addressed to another page (stale token) is ignored.
    internal.receive({ t: "reply", id: 1, k: "stale", ok: true, v: "wrong" });
    internal.receive({ t: "reply", id: 1, k: msg.k, ok: true, v: "hello bun" });
    expect(await p).toBe("hello bun");
  });

  test("rejects with a CallError", async () => {
    const { sent, runtime, internal } = setup();
    const p = runtime.call("Files.Read", "/nope");
    const { id, k } = sent[0] as CallMsg;
    internal.receive([{ t: "reply", id, k, ok: false, e: "file not found" }]);
    const err = (await p.catch((e) => e)) as CallError;
    expect(err).toBeInstanceOf(CallError);
    expect(err.message).toBe("file not found");
    expect(err.method).toBe("Files.Read");
  });

  test("concurrent calls resolve independently", async () => {
    const { sent, runtime, internal } = setup();
    const a = runtime.call("A.Do");
    const b = runtime.call("B.Do");
    const [ma, mb] = sent as CallMsg[];
    internal.receive([
      { t: "reply", id: mb!.id, k: mb!.k, ok: true, v: "b" },
      { t: "reply", id: ma!.id, k: ma!.k, ok: true, v: "a" },
    ]);
    expect(await Promise.all([a, b])).toEqual(["a", "b"]);
  });

  test("window controls call built-in methods", () => {
    const { sent, runtime } = setup();
    runtime.window.minimize();
    runtime.window.setTitle("Hi");
    expect(sent.map((m) => (m as CallMsg).m)).toEqual(["mygo:window.Minimize", "mygo:window.SetTitle"]);
    expect((sent[1] as CallMsg).a).toEqual(["Hi"]);
  });
});

describe("events", () => {
  test("on / once / unsubscribe", () => {
    const { runtime, internal } = setup();
    const seen: unknown[] = [];
    const off = runtime.on("tick", (v) => seen.push(["on", v]));
    runtime.once("tick", (v) => seen.push(["once", v]));

    internal.receive({ t: "event", n: "tick", p: 1 });
    internal.receive({ t: "event", n: "tick", p: 2 });
    expect(seen).toEqual([
      ["on", 1],
      ["once", 1],
      ["on", 2],
    ]);

    off();
    internal.receive({ t: "event", n: "tick", p: 3 });
    expect(seen.length).toBe(3);
  });

  test("the same listener can subscribe twice", () => {
    const { runtime, internal } = setup();
    let n = 0;
    const fn = () => n++;
    const off1 = runtime.on("x", fn);
    runtime.on("x", fn);
    internal.receive({ t: "event", n: "x" });
    off1();
    internal.receive({ t: "event", n: "x" });
    expect(n).toBe(3);
  });
});

describe("channels", () => {
  type AckMsg = Extract<Outgoing, { t: "chan-ack" }>;
  type CloseMsg = Extract<Outgoing, { t: "chan-close" }>;

  test("stream values to onmessage, in order, until Go closes them", async () => {
    const { sent, runtime, internal } = setup();
    const got: unknown[] = [];
    let closed = 0;
    const ch = runtime.channel<number>((v) => got.push(v));
    ch.onclose = () => closed++;
    const p = runtime.call("Svc.Stream", "x", ch);
    const { id, k, a } = sent[0] as CallMsg;
    expect(a).toEqual(["x", 1]); // the channel goes as its id

    internal.receive([
      { t: "chan", c: 1, k, p: 1 },
      { t: "chan", c: 1, k: "stale", p: 99 },
      { t: "chan", c: 1, k, p: 2, a: 1 },
    ]);
    expect(got).toEqual([1, 2]);
    expect(sent[1]).toEqual({ t: "chan-ack", c: 1, k, n: 2 } satisfies AckMsg);

    internal.receive([
      { t: "chan", c: 1, k, end: true },
      { t: "reply", id, k, ok: true },
    ]);
    await p;
    expect(ch.closed).toBe(true);
    expect(closed).toBe(1);
    internal.receive({ t: "chan", c: 1, k, p: 3 }); // after the end
    expect(got).toEqual([1, 2]);
  });

  test("iterate values, including those that arrived first", async () => {
    const { sent, runtime, internal } = setup();
    const ch = runtime.channel<string>();
    runtime.call("Svc.Stream", ch);
    const { k } = sent[0] as CallMsg;
    internal.receive([
      { t: "chan", c: 1, k, p: "a" },
      { t: "chan", c: 1, k, p: "b", a: 1 },
    ]);
    expect(sent.length).toBe(1); // not taken yet: no acknowledgment
    const got: string[] = [];
    const loop = (async () => {
      for await (const v of ch) got.push(v);
    })();
    await Promise.resolve();
    expect(sent[1]).toMatchObject({ t: "chan-ack", n: 2 });
    internal.receive([
      { t: "chan", c: 1, k, p: "c" },
      { t: "chan", c: 1, k, end: true },
    ]);
    await loop;
    expect(got).toEqual(["a", "b", "c"]);
  });

  test("leaving the loop closes the channel", async () => {
    const { sent, runtime, internal } = setup();
    const ch = runtime.channel<number>();
    runtime.call("Svc.Stream", ch);
    const { k } = sent[0] as CallMsg;
    internal.receive([
      { t: "chan", c: 1, k, p: 1 },
      { t: "chan", c: 1, k, p: 2 },
    ]);
    for await (const v of ch) {
      if (v === 1) break;
    }
    expect(ch.closed).toBe(true);
    expect(sent[1]).toEqual({ t: "chan-close", c: 1, k } satisfies CloseMsg);
    // A closed channel can not be passed again.
    await expect(runtime.call("Svc.Stream", ch)).rejects.toThrow("one call");
  });

  test("onmessage and a waiting iterator both get their due", async () => {
    const { sent, runtime, internal } = setup();
    const ch = runtime.channel<number>();
    runtime.call("Svc.Stream", ch);
    const { k } = sent[0] as CallMsg;
    const it = ch[Symbol.asyncIterator]();
    const next = it.next();
    internal.receive({ t: "chan", c: 1, k, p: 1 });
    expect(await next).toEqual({ value: 1, done: false });
    const got: number[] = [];
    ch.onmessage = (v) => got.push(v);
    const pending = it.next();
    internal.receive([
      { t: "chan", c: 1, k, p: 2 },
      { t: "chan", c: 1, k, end: true },
    ]);
    expect(got).toEqual([2]);
    expect(await pending).toEqual({ value: undefined, done: true });
  });

  test("a failed call ends its channels", async () => {
    const { sent, runtime, internal } = setup();
    const ch = runtime.channel();
    const p = runtime.call("Svc.Missing", ch);
    const { id, k } = sent[0] as CallMsg;
    const done = (async () => {
      for await (const _ of ch);
    })();
    internal.receive({ t: "reply", id, k, ok: false, e: "method Svc.Missing is not bound" });
    await expect(p).rejects.toThrow("not bound");
    await done;
    expect(ch.closed).toBe(true);
  });
});

test("runtime object is frozen", () => {
  const { runtime } = setup();
  expect(runtime.platform).toBe("darwin");
  expect(runtime.windowId).toBe(7);
  expect(Object.isFrozen(runtime)).toBe(true);
  expect(Object.isFrozen(runtime.window)).toBe(true);
});

test("preferences are the latest Go told, before listeners hear them", () => {
  const { runtime, internal } = setup();
  expect(runtime.preferences).toEqual({
    accent: null,
    reduceMotion: false,
    reduceTransparency: false,
    highContrast: false,
    textScale: 1,
  });
  const p = { accent: "#ff8000", reduceMotion: true, reduceTransparency: true, highContrast: false, textScale: 1.5 };
  let seen: unknown;
  runtime.on("mygo:preferences", () => (seen = runtime.preferences));
  internal.receive({ t: "event", n: "mygo:preferences", p });
  expect(runtime.preferences).toEqual(p);
  expect(seen).toEqual(p);
});

describe("locale", () => {
  test("comes from the configuration, then from the event", () => {
    const { runtime, internal } = setup();
    expect(runtime.locale).toBe("en");
    expect(runtime.dir).toBe("ltr");
    const seen: unknown[] = [];
    runtime.on("mygo:locale", (p) => seen.push(p));
    internal.receive({ t: "event", n: "mygo:locale", p: { locale: "ar", dir: "rtl" } });
    expect(seen).toEqual([{ locale: "ar", dir: "rtl" }]);
    expect(runtime.locale).toBe("ar");
    expect(runtime.dir).toBe("rtl");
    // A payload that is no language changes nothing.
    internal.receive({ t: "event", n: "mygo:locale", p: 7 });
    expect(runtime.locale).toBe("ar");
    expect(seen.length).toBe(1);
  });

  test("the language the page has tells no listener", () => {
    const { runtime, internal } = setup();
    const seen: string[] = [];
    runtime.on<{ locale: string }>("mygo:locale", (p) => seen.push(p.locale));
    // As the page loads, Go tells it the language its configuration has.
    internal.receive({ t: "event", n: "mygo:locale", p: { locale: "en", dir: "ltr" } });
    expect(seen).toEqual([]);
    internal.receive({ t: "event", n: "mygo:locale", p: { locale: "ja", dir: "ltr" } });
    internal.receive({ t: "event", n: "mygo:locale", p: { locale: "ja", dir: "ltr" } });
    expect(seen).toEqual(["ja"]);
  });
});
