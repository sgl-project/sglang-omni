import type {
  BridgeConfig,
  Channel,
  Direction,
  Incoming,
  LocaleChange,
  Outgoing,
  Runtime,
  ThemePreferences,
} from "./types";

type Pending = {
  method: string;
  resolve: (v: any) => void;
  reject: (e: unknown) => void;
  channels?: ChannelState[];
};
type Listener = (payload: any) => void;

/** The side of a channel that Go messages reach. */
interface ChannelState {
  /** Set when the channel is passed to a call. */
  id: number;
  readonly closed: boolean;
  /** Delivers a value; ask requests an acknowledgment once it is taken. */
  push(value: unknown, ask: boolean): void;
  /** Go closed the channel. */
  end(): void;
}

/** Reports an error thrown by page code without stopping the runtime. */
const report = (err: unknown) =>
  setTimeout(() => {
    throw err;
  });

/** Internal API called by the Go side through `window.__mygo`. */
export interface Internal {
  /** Delivers one or more messages from Go. */
  receive(messages: Incoming | Incoming[]): void;
}

/** Error thrown by `call` when the Go method returns an error. */
export class CallError extends Error {
  constructor(
    readonly method: string,
    message: string,
  ) {
    super(message);
    this.name = "CallError";
  }
}

/**
 * Creates the renderer runtime. `post` delivers a serialized message to the
 * Go side. Kept free of DOM access so it can be unit tested.
 */
export function createRuntime(
  config: BridgeConfig,
  post: (message: string) => void,
): { runtime: Runtime; internal: Internal } {
  // A per-page token keeps replies addressed to a previous page (before a
  // navigation) from resolving promises of the current one.
  const token = Math.random().toString(36).slice(2) + Date.now().toString(36);
  const pending = new Map<number, Pending>();
  const listeners = new Map<string, Set<Listener>>();
  // The desktop's preferences, which Go tells at document start and on
  // every change; the defaults stand until then.
  let preferences: ThemePreferences = {
    accent: null,
    reduceMotion: false,
    reduceTransparency: false,
    highContrast: false,
    textScale: 1,
  };
  let seq = 0;
  // Channels passed to calls, by id, and the state of every channel.
  const channels = new Map<number, ChannelState>();
  const states = new WeakMap<object, ChannelState>();
  let channelSeq = 0;
  // The language the app shows: the configuration tells it at document
  // start, and the mygo:locale event when it changes.
  let locale = config.locale || "en";
  let dir: Direction = config.dir === "rtl" ? "rtl" : "ltr";

  const send = (message: Outgoing) => post(JSON.stringify(message));

  const call = <T>(method: string, ...args: unknown[]): Promise<T> => {
    const id = ++seq;
    return new Promise<T>((resolve, reject) => {
      // Channels among the arguments stream values back.
      let chans: ChannelState[] | undefined;
      for (const arg of args) {
        const ch = typeof arg === "object" && arg !== null ? states.get(arg) : undefined;
        if (!ch) continue;
        if (ch.id || ch.closed) {
          chans?.forEach((c) => channels.delete(c.id));
          reject(new Error("mygo: a channel can be passed to one call, before it is closed"));
          return;
        }
        ch.id = ++channelSeq;
        channels.set(ch.id, ch);
        (chans ??= []).push(ch);
      }
      pending.set(id, { method, resolve, reject, channels: chans });
      try {
        send({ t: "call", id, k: token, m: method, a: args });
      } catch (err) {
        pending.delete(id);
        chans?.forEach((c) => channels.delete(c.id));
        reject(err);
      }
    });
  };

  const channel = <T>(onmessage?: (value: T) => void): Channel<T> => {
    let handler: ((value: T) => void) | null = null;
    let closed = false;
    // Values not taken yet, and next() calls waiting for one.
    let queue: { value: T; n: number; ask: boolean }[] = [];
    const waiting: ((result: IteratorResult<T>) => void)[] = [];
    let received = 0;

    // Tells Go the page took the values up to the n-th.
    const ack = (n: number) => {
      try {
        send({ t: "chan-ack", c: state.id, k: token, n });
      } catch {
        // The page is being torn down.
      }
    };
    const handle = (value: T) => {
      try {
        handler!(value);
      } catch (err) {
        report(err);
      }
    };
    const finish = () => {
      channels.delete(state.id);
      if (queue.length === 0) waiting.splice(0).forEach((w) => w({ value: undefined, done: true }));
      if (ch.onclose) {
        try {
          ch.onclose();
        } catch (err) {
          report(err);
        }
      }
    };

    const state: ChannelState = {
      id: 0,
      get closed() {
        return closed;
      },
      push(value, ask) {
        if (closed) return;
        const n = ++received;
        if (handler) handle(value as T);
        else if (waiting.length > 0) waiting.shift()!({ value: value as T, done: false });
        else return void queue.push({ value: value as T, n, ask });
        if (ask) ack(n);
      },
      end() {
        if (closed) return;
        closed = true;
        finish(); // values not taken yet can still be iterated
      },
    };

    const ch = {
      get onmessage() {
        return handler;
      },
      set onmessage(fn: ((value: T) => void) | null) {
        handler = fn;
        // Values that arrived earlier come first.
        while (handler && queue.length > 0) {
          const q = queue.shift()!;
          handle(q.value);
          if (q.ask) ack(q.n);
        }
      },
      onclose: null as (() => void) | null,
      get closed() {
        return closed;
      },
      close() {
        if (closed) return;
        closed = true;
        queue = [];
        if (state.id) {
          try {
            send({ t: "chan-close", c: state.id, k: token });
          } catch {
            // The page is being torn down.
          }
        }
        finish();
      },
      [Symbol.asyncIterator](): AsyncIterator<T> {
        return {
          next() {
            const q = queue.shift();
            if (q) {
              if (q.ask) ack(q.n);
              return Promise.resolve({ value: q.value, done: false });
            }
            if (closed) return Promise.resolve({ value: undefined, done: true });
            return new Promise((resolve) => waiting.push(resolve));
          },
          // Leaving a for await loop early stops the stream.
          return() {
            ch.close();
            return Promise.resolve({ value: undefined, done: true });
          },
        };
      },
      // Goes to Go as its id.
      toJSON: () => state.id,
    };
    states.set(ch, state);
    if (onmessage) ch.onmessage = onmessage;
    return ch;
  };

  const on = (event: string, listener: Listener) => {
    if (typeof listener !== "function") throw new TypeError("listener must be a function");
    let set = listeners.get(event);
    if (!set) listeners.set(event, (set = new Set()));
    // Wrap so the same function can be subscribed twice and removed once.
    const entry: Listener = (p) => listener(p);
    set.add(entry);
    return () => {
      set.delete(entry);
      if (set.size === 0 && listeners.get(event) === set) listeners.delete(event);
    };
  };

  const once = (event: string, listener: Listener) => {
    const off = on(event, (p) => {
      off();
      listener(p);
    });
    return off;
  };

  const internal: Internal = {
    receive(messages) {
      for (const msg of Array.isArray(messages) ? messages : [messages]) {
        if (msg.t === "event") {
          // Before the listeners, which may read it.
          if (msg.n === "mygo:preferences") preferences = Object.freeze(msg.p as ThemePreferences);
          if (msg.n === "mygo:locale") {
            const p = msg.p as Partial<LocaleChange> | undefined;
            if (typeof p?.locale !== "string") continue;
            // Go tells a page the language again as it loads, for one that
            // started loading as it changed: listeners hear of a language
            // the page did not have only.
            if (p.locale === locale) continue;
            locale = p.locale;
            dir = p.dir === "rtl" ? "rtl" : "ltr";
          }
          const set = listeners.get(msg.n);
          if (!set) continue;
          for (const listener of [...set]) {
            try {
              listener(msg.p);
            } catch (err) {
              // One failing listener must not break the others.
              report(err);
            }
          }
        } else if (msg.t === "chan") {
          if (msg.k !== token) continue;
          const ch = channels.get(msg.c);
          if (!ch) continue;
          if (msg.end) ch.end();
          else ch.push(msg.p, msg.a === 1);
        } else if (msg.t === "reply") {
          if (msg.k !== token) continue;
          const p = pending.get(msg.id);
          if (!p) continue;
          pending.delete(msg.id);
          // Go closed the call's channels before replying, unless the call
          // failed before they were made.
          p.channels?.forEach((c) => c.end());
          if (msg.ok) p.resolve(msg.v);
          else p.reject(new CallError(p.method, msg.e ?? "unknown error"));
        }
      }
    },
  };

  const win = (op: string) => () => call<any>(`mygo:window.${op}`);
  const runtime: Runtime = Object.freeze({
    call,
    channel,
    on,
    once,
    platform: config.platform,
    windowId: config.windowId,
    version: config.version,
    get preferences() {
      return preferences;
    },
    get locale() {
      return locale;
    },
    get dir() {
      return dir;
    },
    window: Object.freeze({
      minimize: win("Minimize"),
      maximize: win("Maximize"),
      unmaximize: win("Unmaximize"),
      toggleMaximize: win("ToggleMaximize"),
      isMaximized: win("IsMaximized"),
      toggleFullScreen: win("ToggleFullScreen"),
      close: win("Close"),
      setTitle: (title: string) => call<void>("mygo:window.SetTitle", title),
    }),
  });

  return { runtime, internal };
}
