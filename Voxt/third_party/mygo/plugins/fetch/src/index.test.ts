import { afterEach, expect, test } from "bun:test";
import { b64, fakeGo, tick, type GoCall } from "../../fake-go";
import { fetch } from "./index";

let go: ReturnType<typeof fakeGo> | undefined;
afterEach(() => go?.uninstall());

interface Req {
  url: string;
  method: string;
  headers: [string, string][];
  body: string | null;
  redirect: string;
}

const head = (status = 200, headers: [string, string][] = []) => ({
  status,
  statusText: "OK",
  url: "https://example.com/final",
  redirected: true,
  headers,
});

test("sends the request and streams the response", async () => {
  let got: Req | undefined;
  go = fakeGo({
    "plugin:fetch.Fetch": (c: GoCall) => {
      got = c.args[0] as Req;
      c.send(1, head(201, [["content-type", "application/json"], ["x-a", "1"]]));
      c.send(1, { chunk: b64('{"a":') });
      c.send(1, { chunk: b64("1}") });
    },
  });
  const res = await fetch(new URL("https://example.com/items"), {
    method: "post",
    headers: { origin: "https://app.test", "x-token": "t" },
    body: new URLSearchParams({ q: "go" }),
  });
  expect(got).toMatchObject({ url: "https://example.com/items", method: "POST", redirect: "follow" });
  expect(Object.fromEntries(got!.headers)).toEqual({
    origin: "https://app.test",
    "x-token": "t",
    "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
  });
  expect(Buffer.from(got!.body!, "base64").toString()).toBe("q=go");
  expect(res).toBeInstanceOf(Response);
  expect(res.status).toBe(201);
  expect(res.url).toBe("https://example.com/final");
  expect(res.redirected).toBe(true);
  expect(res.headers.get("x-a")).toBe("1");
  expect(await res.json()).toEqual({ a: 1 });
});

test("takes a Request", async () => {
  let got: Req | undefined;
  go = fakeGo({
    "plugin:fetch.Fetch": (c) => {
      got = c.args[0] as Req;
      c.send(1, head(204));
    },
  });
  const res = await fetch(new Request("https://example.com/x", { method: "PUT", body: "hi", redirect: "manual" }));
  expect(got).toMatchObject({ method: "PUT", redirect: "manual", body: b64("hi") });
  expect(res.status).toBe(204);
  expect(res.body).toBeNull();
});

test("rejects when Go fails", async () => {
  go = fakeGo({
    "plugin:fetch.Fetch": () => {
      throw new Error("dial tcp: connection refused");
    },
  });
  const err = await fetch("https://example.com").catch((e) => e);
  expect(err).toBeInstanceOf(TypeError);
  expect(err.message).toBe("fetch failed: dial tcp: connection refused");
});

test("explains a missing Go side", async () => {
  go = fakeGo({});
  const err = await fetch("https://example.com").catch((e) => e);
  expect(err.message).toBe("fetch failed: method plugin:fetch.Fetch is not bound");
  go.uninstall();
  go = fakeGo({
    "plugin:fetch.Fetch": () => {
      throw new Error("the fetch plugin is not used: call mygo.Use with it before App.Run");
    },
  });
  expect((await fetch("https://example.com").catch((e) => e)).message).toContain("mygo.Use(fetch.Plugin)");
});

test("errors the body when Go fails while streaming", async () => {
  go = fakeGo({
    "plugin:fetch.Fetch": (c) => {
      c.send(1, head());
      c.send(1, { chunk: b64("par") });
      throw new Error("unexpected EOF");
    },
  });
  const res = await fetch("https://example.com");
  expect((await res.text().catch((e) => e)).message).toBe("fetch failed: unexpected EOF");
});

test("aborts", async () => {
  let stopped = false;
  const streaming = (c: GoCall) =>
    new Promise<void>((resolve) => {
      c.send(1, head());
      c.send(1, { chunk: b64("first") });
      c.signal.addEventListener("abort", () => {
        stopped = true;
        resolve();
      });
    });
  go = fakeGo({ "plugin:fetch.Fetch": streaming });

  const ctrl = new AbortController();
  const res = await fetch("https://example.com", { signal: ctrl.signal });
  const reader = res.body!.getReader();
  expect(new TextDecoder().decode((await reader.read()).value)).toBe("first");
  ctrl.abort();
  expect((await reader.read().catch((e) => e)).name).toBe("AbortError");
  await tick();
  expect(stopped).toBe(true);

  // Before the response.
  go.uninstall();
  go = fakeGo({ "plugin:fetch.Fetch": (c) => new Promise<void>((r) => c.signal.addEventListener("abort", () => r())) });
  const early = new AbortController();
  const p = fetch("https://example.com", { signal: early.signal });
  await tick();
  early.abort();
  expect((await p.catch((e) => e)).name).toBe("AbortError");
  expect((await fetch("https://example.com", { signal: AbortSignal.abort() }).catch((e) => e)).name).toBe("AbortError");

  // Canceling the body stops Go too.
  go.uninstall();
  stopped = false;
  go = fakeGo({ "plugin:fetch.Fetch": streaming });
  await (await fetch("https://example.com")).body!.cancel();
  await tick();
  expect(stopped).toBe(true);
});

test("leaves other schemes to the webview", async () => {
  go = fakeGo({});
  const res = await fetch("data:text/plain,hello");
  expect(await res.text()).toBe("hello");
  expect(go.calls).toEqual([]);
});

test("refuses a GET body", async () => {
  go = fakeGo({});
  expect(fetch("https://example.com", { body: "x" })).rejects.toThrow(TypeError);
});
