import { afterEach, expect, test } from "bun:test";
import {
  Channel,
  call,
  currentWindow,
  direction,
  event,
  isCallError,
  isMyGo,
  locale,
  on,
  onDocumentReload,
  onDocumentState,
  onFileDrop,
  onLocaleChange,
  onPreferences,
  preferences,
  runtime,
  type DocumentState,
  type FileDrop,
  type Runtime,
  type ThemePreferences,
} from "./index";

const calls: [string, unknown[]][] = [];
const listeners = new Map<string, (payload: unknown) => void>();

const fake: Runtime = {
  call: async <T>(method: string, ...args: unknown[]) => {
    calls.push([method, args]);
    if (method === "Svc.Fail") {
      const err = new Error("boom") as Error & { method: string };
      err.name = "CallError";
      err.method = method;
      throw err;
    }
    return method as T;
  },
  channel: <T>(onmessage?: (value: T) => void) =>
    ({ onmessage, onclose: null, closed: false, close() {}, [Symbol.asyncIterator]: () => ({}) }) as unknown as Channel<T>,
  on: (name, listener) => {
    listeners.set(name, listener as (payload: unknown) => void);
    return () => listeners.delete(name);
  },
  once: (name, listener) => {
    listeners.set(name, listener as (payload: unknown) => void);
    return () => listeners.delete(name);
  },
  platform: "darwin",
  windowId: 1,
  version: "0.1.0",
  preferences: { accent: null, reduceMotion: false, reduceTransparency: true, highContrast: false, textScale: 1 },
  locale: "en",
  dir: "ltr",
  window: {
    minimize: () => fake.call("mygo:window.Minimize"),
    maximize: () => fake.call("mygo:window.Maximize"),
    unmaximize: () => fake.call("mygo:window.Unmaximize"),
    toggleMaximize: () => fake.call("mygo:window.ToggleMaximize"),
    isMaximized: () => fake.call("mygo:window.IsMaximized"),
    toggleFullScreen: () => fake.call("mygo:window.ToggleFullScreen"),
    close: () => fake.call("mygo:window.Close"),
    setTitle: (title) => fake.call("mygo:window.SetTitle", title),
  },
};

afterEach(() => {
  delete (globalThis as { mygo?: Runtime }).mygo;
  calls.length = 0;
  listeners.clear();
});

test("outside MyGo", async () => {
  expect(isMyGo()).toBe(false);
  expect(() => runtime()).toThrow("runtime not found");
  await expect(call("Svc.Method")).rejects.toThrow("runtime not found");
});

test("delegates to window.mygo", async () => {
  (globalThis as { mygo?: Runtime }).mygo = fake;
  expect(isMyGo()).toBe(true);
  expect(await call<string>("Svc.Method", 1, "a")).toBe("Svc.Method");
  expect(calls).toEqual([["Svc.Method", [1, "a"]]]);

  const got: unknown[] = [];
  const off = event<{ n: number }>("tick").on((p) => got.push(p.n));
  listeners.get("tick")?.({ n: 7 });
  off();
  expect(got).toEqual([7]);
  expect(listeners.has("tick")).toBe(false);

  on("other", () => {});
  expect(listeners.has("other")).toBe(true);

  await currentWindow.setTitle("Hi");
  expect(calls.at(-1)).toEqual(["mygo:window.SetTitle", ["Hi"]]);

  const err = await call("Svc.Fail").catch((e: unknown) => e);
  expect(isCallError(err)).toBe(true);
  expect(isCallError(new Error("other"))).toBe(false);
});

test("Channel", () => {
  expect(() => new Channel()).toThrow("runtime not found");
  (globalThis as { mygo?: Runtime }).mygo = fake;
  const fn = (n: number) => n;
  const ch = new Channel<number>(fn);
  expect(ch.onmessage).toBe(fn);
  expect(ch.closed).toBe(false);
});

test("onFileDrop", () => {
  (globalThis as { mygo?: Runtime }).mygo = fake;
  const got: FileDrop[] = [];
  const off = onFileDrop((d) => got.push(d));
  listeners.get("mygo:file-drop")?.({ paths: ["/tmp/a.txt"], x: 10, y: 20 });
  expect(got).toEqual([{ paths: ["/tmp/a.txt"], x: 10, y: 20 }]);
  off();
  expect(listeners.has("mygo:file-drop")).toBe(false);
});

test("onDocumentState and onDocumentReload", () => {
  (globalThis as { mygo?: Runtime }).mygo = fake;
  const states: DocumentState[] = [];
  let reloads = 0;
  const off = onDocumentState((s) => states.push(s));
  const offReload = onDocumentReload(() => reloads++);
  const state = { name: "a.md", path: "/tmp/a.md", edited: true, readOnly: false, conflict: false, deleted: false };
  listeners.get("mygo:document-state")?.(state);
  listeners.get("mygo:document-reload")?.({});
  expect(states).toEqual([state]);
  expect(reloads).toBe(1);
  off();
  offReload();
  expect(listeners.has("mygo:document-state")).toBe(false);
  expect(listeners.has("mygo:document-reload")).toBe(false);
});

test("preferences and onPreferences", () => {
  (globalThis as { mygo?: Runtime }).mygo = fake;
  expect(preferences()).toEqual(fake.preferences);
  const got: ThemePreferences[] = [];
  const off = onPreferences((p) => got.push(p));
  const changed = { accent: "#ff8000", reduceMotion: true, reduceTransparency: false, highContrast: false, textScale: 1.25 };
  listeners.get("mygo:preferences")?.(changed);
  expect(got).toEqual([changed]);
  off();
  expect(listeners.has("mygo:preferences")).toBe(false);
});

test("locale, direction and onLocaleChange", () => {
  (globalThis as { mygo?: Runtime }).mygo = fake;
  expect(locale()).toBe("en");
  expect(direction()).toBe("ltr");
  const got: [string, string][] = [];
  const off = onLocaleChange((tag, dir) => got.push([tag, dir]));
  listeners.get("mygo:locale")?.({ locale: "ar", dir: "rtl" });
  expect(got).toEqual([["ar", "rtl"]]);
  off();
  expect(listeners.has("mygo:locale")).toBe(false);
});
