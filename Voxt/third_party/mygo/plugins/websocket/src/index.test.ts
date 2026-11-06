import { afterEach, expect, test } from "bun:test";
import { b64, fakeGo, tick, type GoCall } from "../../fake-go";
import { WebSocket } from "./index";

let go: ReturnType<typeof fakeGo> | undefined;
afterEach(() => go?.uninstall());

/** A server that echoes, as the Go side streams it. */
function echoServer() {
  const sent: unknown[][] = [];
  let conn: GoCall | undefined;
  let finish: (() => void) | undefined;
  go = fakeGo({
    "plugin:websocket.Connect": (c) =>
      new Promise<void>((resolve) => {
        conn = c;
        finish = resolve;
        c.signal.addEventListener("abort", () => resolve());
        c.send(1, { type: "open", protocol: "chat" });
      }),
    "plugin:websocket.Send": (c) => {
      sent.push(c.args);
      const m = c.args[2] as { type: string; text?: string; data?: string; code?: number };
      if (m.type === "close") {
        conn!.send(1, { type: "close", code: m.code, reason: "bye", clean: true });
        finish!();
      } else conn!.send(1, m.type === "text" ? { type: "text", text: m.text } : { type: "binary", data: m.data });
    },
  });
  return { sent, server: () => conn! };
}

const next = (ws: WebSocket, type: string) => new Promise<any>((r) => ws.addEventListener(type, r, { once: true }));

test("opens, echoes and closes", async () => {
  const { sent } = echoServer();
  const ws = new WebSocket("https://example.com/live", ["chat"], { headers: { authorization: "Bearer t" } });
  expect(ws.url).toBe("wss://example.com/live");
  expect(ws.readyState).toBe(WebSocket.CONNECTING);
  expect(() => ws.send("early")).toThrow("not open");
  await next(ws, "open");
  expect(ws.readyState).toBe(ws.OPEN);
  expect(ws.protocol).toBe("chat");
  const [connect] = go!.calls[0]![1] as [{ id: string; headers: [string, string][] }];
  expect(connect.headers).toEqual([["authorization", "Bearer t"]]);

  const got: unknown[] = [];
  ws.onmessage = (e) => got.push(e.data);
  ws.send("hello");
  expect(ws.bufferedAmount).toBe(5);
  ws.binaryType = "arraybuffer";
  ws.send(new Uint8Array([1, 2, 3]));
  while (got.length < 2 || ws.bufferedAmount > 0) await tick();
  expect(got[0]).toBe("hello");
  expect([...new Uint8Array(got[1] as ArrayBuffer)]).toEqual([1, 2, 3]);
  // Messages carry their order.
  expect(sent.map((s) => s[1])).toEqual([1, 2]);
  expect(sent[0]![0]).toBe(connect.id);

  const closed = next(ws, "close");
  let errored = false;
  ws.onerror = () => (errored = true);
  ws.close(4001, "done");
  expect(ws.readyState).toBe(WebSocket.CLOSING);
  const e = await closed;
  expect([e.code, e.reason, e.wasClean]).toEqual([4001, "bye", true]);
  expect(ws.readyState).toBe(WebSocket.CLOSED);
  expect(errored).toBe(false);
  expect(sent[2]![2]).toEqual({ type: "close", code: 4001, reason: "done" });
});

test("receives blobs", async () => {
  const { server } = echoServer();
  const ws = new WebSocket("ws://example.com");
  await next(ws, "open");
  const msg = next(ws, "message");
  server().send(1, { type: "binary", data: b64("blob") });
  const e = await msg;
  expect(e.data).toBeInstanceOf(Blob);
  expect(await e.data.text()).toBe("blob");
});

test("fails", async () => {
  go = fakeGo({
    "plugin:websocket.Connect": (c) => c.send(1, { type: "close", code: 1006, error: "the server answered the handshake with 403 Forbidden" }),
  });
  const errors: string[] = [];
  const log = console.error;
  console.error = (m: string) => errors.push(m);
  try {
    const ws = new WebSocket("ws://example.com");
    const order: string[] = [];
    ws.onerror = () => order.push("error");
    ws.onclose = (e) => order.push(`close ${e.code} ${e.wasClean}`);
    await next(ws, "close");
    expect(order).toEqual(["error", "close 1006 false"]);
    expect(errors[0]).toContain("403 Forbidden");
  } finally {
    console.error = log;
  }
});

test("closing while connecting stops the handshake", async () => {
  let stopped = false;
  go = fakeGo({
    "plugin:websocket.Connect": (c) =>
      new Promise<void>((r) =>
        c.signal.addEventListener("abort", () => {
          stopped = true;
          r();
        }),
      ),
  });
  const log = console.error;
  console.error = () => {};
  try {
    const ws = new WebSocket("ws://example.com");
    await tick();
    ws.close();
    const e = await next(ws, "close");
    expect(e.code).toBe(1006);
    expect(stopped).toBe(true);
  } finally {
    console.error = log;
  }
});

test("validates", () => {
  go = fakeGo({});
  expect(() => new WebSocket("ftp://example.com")).toThrow("scheme");
  expect(() => new WebSocket("ws://example.com/#x")).toThrow("fragment");
  expect(() => new WebSocket("ws://example.com", ["a", "a"])).toThrow("duplicates");
  const ws = new WebSocket("ws://example.com");
  expect(() => ws.close(1001)).toThrow("Invalid close code");
  expect(() => ws.close(1000, "x".repeat(124))).toThrow("123 bytes");
});
