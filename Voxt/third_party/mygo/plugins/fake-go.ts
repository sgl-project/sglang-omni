// A Go side for the plugins' tests: the real page runtime of the bridge,
// installed as window.mygo, whose calls go to handlers written in
// TypeScript that stream through channels as Go methods do.
import { createRuntime } from "../packages/bridge/src/runtime";
import type { Outgoing } from "../packages/bridge/src/types";

/** A call being handled. */
export interface GoCall {
  args: unknown[];
  /** Sends a value on the channel passed as the i-th argument. */
  send(i: number, value: unknown): void;
  /** Aborted when the page closes a channel of the call. */
  signal: AbortSignal;
}

export type Handler = (call: GoCall) => unknown;

/** Methods whose argument 1 is a channel, as those of the plugins. */
const streaming = new Set(["plugin:fetch.Fetch", "plugin:websocket.Connect", "plugin:watch.Watch"]);

/** Installs window.mygo with handlers for methods; returns what was called. */
export function fakeGo(handlers: Record<string, Handler>): { calls: [string, unknown[]][]; uninstall(): void } {
  const calls: [string, unknown[]][] = [];
  const aborts = new Map<number, AbortController>();
  const { runtime, internal } = createRuntime(
    { platform: "darwin", windowId: 1, version: "0.0.0-test", secret: "s", locale: "en", dir: "ltr" },
    (raw) => {
      const m = JSON.parse(raw) as Outgoing;
      if (m.t === "chan-close") aborts.get(m.c)?.abort();
      if (m.t !== "call") return;
      calls.push([m.m, m.a]);
      const abort = new AbortController();
      const chans = m.a.map((a, i) => (streaming.has(m.m) && i === 1 && handlers[m.m] ? (a as number) : 0));
      chans.forEach((c) => c && aborts.set(c, abort));
      // Like Go: a goroutine per call, the channels end before the reply.
      setTimeout(async () => {
        const handler = handlers[m.m];
        let reply: { ok: boolean; v?: unknown; e?: string };
        try {
          if (!handler) throw new Error(`method ${m.m} is not bound`);
          const v = await handler({
            args: m.a,
            send(i, value) {
              if (abort.signal.aborted) throw new Error("mygo: channel closed");
              internal.receive({ t: "chan", c: chans[i]!, k: m.k, p: value });
            },
            signal: abort.signal,
          });
          reply = { ok: true, v };
        } catch (err) {
          reply = { ok: false, e: err instanceof Error ? err.message : String(err) };
        }
        for (const c of chans) if (c) internal.receive({ t: "chan", c, k: m.k, end: true });
        internal.receive({ t: "reply", id: m.id, k: m.k, ...reply });
      });
    },
  );
  (globalThis as { mygo?: unknown }).mygo = runtime;
  return { calls, uninstall: () => delete (globalThis as { mygo?: unknown }).mygo };
}

export const tick = () => new Promise((r) => setTimeout(r, 1));

export const b64 = (s: string) => Buffer.from(s).toString("base64");
