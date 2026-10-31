/**
 * The WebSocket plugin of MyGo: a `WebSocket` whose connection Go makes,
 * with the API of the browser's. Unlike the browser's, it may send any
 * header with the opening handshake, such as Authorization or Cookie, and
 * is not subject to the page's Content Security Policy. The Go side must
 * use the plugin:
 *
 * ```go
 * mygo.Use(websocket.Plugin) // github.com/egoist/mygo/plugins/websocket
 * ```
 *
 * ```ts
 * import { WebSocket } from "@mygo-plugins/websocket";
 *
 * const ws = new WebSocket("wss://example.com/live", [], {
 *   headers: { authorization: `Bearer ${token}` },
 * });
 * ws.onmessage = (e) => console.log(e.data);
 * ```
 *
 * @module
 */
import { Channel, call, isCallError } from "mygo-runtime";

/** Options of a {@link WebSocket}, beyond the browser's. */
export interface WebSocketOptions {
  /** Headers of the opening handshake, e.g. Authorization or Origin. */
  headers?: HeadersInit;
}

/** What happens to the connection, from Go. */
interface GoEvent {
  type: "open" | "text" | "binary" | "close";
  protocol?: string;
  text?: string;
  /** Base64. */
  data?: string;
  code?: number;
  reason?: string;
  clean?: boolean;
  error?: string;
}

/** A message for Go. */
type GoMessage =
  | { type: "text"; text: string }
  | { type: "binary"; data: string }
  | { type: "close"; code: number; reason: string }
  | { type: "skip" };

export interface WebSocketEventMap {
  open: Event;
  message: MessageEvent;
  error: Event;
  close: CloseEvent;
}

type Handler<E> = ((this: WebSocket, ev: E) => unknown) | null;

const CONNECTING = 0;
const OPEN = 1;
const CLOSING = 2;
const CLOSED = 3;

const encoder = new TextEncoder();

/**
 * A WebSocket connection made by Go, with the API of the browser's
 * `WebSocket`. The third argument sets headers of the opening handshake.
 */
export class WebSocket extends EventTarget {
  static readonly CONNECTING = CONNECTING;
  static readonly OPEN = OPEN;
  static readonly CLOSING = CLOSING;
  static readonly CLOSED = CLOSED;
  declare readonly CONNECTING: 0;
  declare readonly OPEN: 1;
  declare readonly CLOSING: 2;
  declare readonly CLOSED: 3;

  /** The URL of the connection, with a ws: or wss: scheme. */
  readonly url: string;
  /** How binary messages arrive: as a `Blob`, or an `ArrayBuffer`. */
  binaryType: BinaryType = "blob";

  #id: string;
  #state: number = CONNECTING;
  #protocol = "";
  #buffered = 0;
  #seq = 0;
  /** Why the connection failed, when the page failed it. */
  #failed: string | undefined;
  #events: Channel<GoEvent>;
  #handlers = new Map<string, { fn: Handler<any> }>();

  constructor(url: string | URL, protocols: string | string[] = [], options: WebSocketOptions = {}) {
    super();
    const base = typeof location !== "undefined" ? location.href : undefined;
    let u: URL;
    try {
      u = new URL(url, base);
    } catch {
      throw new DOMException(`Invalid URL: ${url}`, "SyntaxError");
    }
    if (u.protocol === "http:") u.protocol = "ws:";
    else if (u.protocol === "https:") u.protocol = "wss:";
    if (u.protocol !== "ws:" && u.protocol !== "wss:") {
      throw new DOMException(`The URL's scheme must be ws or wss, not ${u.protocol.slice(0, -1)}`, "SyntaxError");
    }
    if (u.hash) throw new DOMException("The URL contains a fragment", "SyntaxError");
    const list = typeof protocols === "string" ? [protocols] : [...protocols];
    if (new Set(list).size !== list.length) {
      throw new DOMException("The subprotocols contain duplicates", "SyntaxError");
    }
    this.url = u.href;
    this.#id = globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2) + Date.now().toString(36);
    this.#events = new Channel<GoEvent>((e) => this.#receive(e));
    call("plugin:websocket.Connect", {
      id: this.#id,
      url: this.url,
      protocols: list,
      headers: [...new Headers(options.headers)],
    }, this.#events).then(
      () => this.#finish(1006, "", false, this.#failed ?? "connection lost"),
      (err: unknown) => this.#finish(1006, "", false, this.#failed ?? failure(err)),
    );
  }

  /** The state of the connection: CONNECTING, OPEN, CLOSING or CLOSED. */
  get readyState(): number {
    return this.#state;
  }

  /** The subprotocol the server chose, once open. */
  get protocol(): string {
    return this.#protocol;
  }

  /** The extensions in use: none. */
  get extensions(): string {
    return "";
  }

  /** Bytes sent with `send` that Go has not written yet. */
  get bufferedAmount(): number {
    return this.#buffered;
  }

  get onopen(): Handler<Event> {
    return this.#handler("open");
  }
  set onopen(fn: Handler<Event>) {
    this.#setHandler("open", fn);
  }
  get onmessage(): Handler<MessageEvent> {
    return this.#handler("message");
  }
  set onmessage(fn: Handler<MessageEvent>) {
    this.#setHandler("message", fn);
  }
  get onerror(): Handler<Event> {
    return this.#handler("error");
  }
  set onerror(fn: Handler<Event>) {
    this.#setHandler("error", fn);
  }
  get onclose(): Handler<CloseEvent> {
    return this.#handler("close");
  }
  set onclose(fn: Handler<CloseEvent>) {
    this.#setHandler("close", fn);
  }

  /** Sends a message: text, or binary data. Throws while connecting. */
  send(data: string | ArrayBufferLike | ArrayBufferView | Blob): void {
    if (this.#state === CONNECTING) throw new DOMException("The connection is not open yet", "InvalidStateError");
    let size: number;
    let message: GoMessage | Promise<GoMessage>;
    if (typeof data === "string") {
      const bytes = encoder.encode(data);
      size = bytes.length;
      message = { type: "text", text: data };
    } else if (typeof Blob !== "undefined" && data instanceof Blob) {
      size = data.size;
      message = data.arrayBuffer().then(
        (b): GoMessage => ({ type: "binary", data: toBase64(new Uint8Array(b)) }),
        (): GoMessage => ({ type: "skip" }),
      );
    } else {
      const bytes = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : new Uint8Array(data as ArrayBufferLike);
      size = bytes.length;
      message = { type: "binary", data: toBase64(bytes) };
    }
    this.#buffered += size;
    if (this.#state !== OPEN) return; // discarded, as browsers do
    const settle = () => (this.#buffered -= size);
    this.#send(message).then(settle, settle);
  }

  /**
   * Closes the connection with a code (1000, or 3000 to 4999) and a reason
   * of at most 123 bytes.
   */
  close(code?: number, reason?: string): void {
    if (code !== undefined && code !== 1000 && !(code >= 3000 && code <= 4999)) {
      throw new DOMException(`Invalid close code ${code}`, "InvalidAccessError");
    }
    const why = reason ?? "";
    if (encoder.encode(why).length > 123) throw new DOMException("The close reason is over 123 bytes", "SyntaxError");
    if (this.#state === CLOSING || this.#state === CLOSED) return;
    if (this.#state === CONNECTING) {
      this.#state = CLOSING;
      this.#failed = "closed before the connection was established";
      this.#events.close(); // stops the handshake
      setTimeout(() => this.#finish(1006, "", false, this.#failed));
      return;
    }
    this.#state = CLOSING;
    this.#send({ type: "close", code: code ?? (why ? 1000 : 1005), reason: why }).catch(() => {});
  }

  #send(message: GoMessage | Promise<GoMessage>): Promise<void> {
    const seq = ++this.#seq;
    return Promise.resolve(message).then((m) => call<void>("plugin:websocket.Send", this.#id, seq, m));
  }

  #receive(e: GoEvent): void {
    switch (e.type) {
      case "open":
        if (this.#state !== CONNECTING) return;
        this.#state = OPEN;
        this.#protocol = e.protocol ?? "";
        this.dispatchEvent(new Event("open"));
        return;
      case "text":
      case "binary": {
        if (this.#state !== OPEN) return;
        let data: unknown = e.text ?? "";
        if (e.type === "binary") {
          const bytes = fromBase64(e.data ?? "");
          data = this.binaryType === "arraybuffer" ? bytes.buffer : new Blob([bytes as Uint8Array<ArrayBuffer>]);
        }
        this.dispatchEvent(new MessageEvent("message", { data, origin: new URL(this.url).origin }));
        return;
      }
      case "close":
        this.#finish(e.code ?? 1005, e.reason ?? "", e.clean ?? false, e.error);
    }
  }

  #finish(code: number, reason: string, wasClean: boolean, error?: string): void {
    if (this.#state === CLOSED) return;
    this.#state = CLOSED;
    this.#events.close();
    if (error) {
      console.error(`WebSocket connection to '${this.url}' failed: ${error}`);
      this.dispatchEvent(new Event("error"));
    }
    this.dispatchEvent(closeEvent(code, reason, wasClean));
  }

  #handler(type: string): Handler<any> {
    return this.#handlers.get(type)?.fn ?? null;
  }

  // An event handler is a listener added when it is first set, which calls
  // whatever handler is set when the event fires.
  #setHandler(type: string, fn: Handler<any>): void {
    let entry = this.#handlers.get(type);
    if (!entry) {
      const e = (entry = { fn: null as Handler<any> });
      this.#handlers.set(type, e);
      this.addEventListener(type, (ev) => e.fn?.call(this, ev));
    }
    entry.fn = typeof fn === "function" ? fn : null;
  }
}

for (const [name, value] of Object.entries({ CONNECTING, OPEN, CLOSING, CLOSED })) {
  Object.defineProperty(WebSocket.prototype, name, { value, enumerable: true });
}

export interface WebSocket {
  addEventListener<K extends keyof WebSocketEventMap>(
    type: K,
    listener: (this: WebSocket, ev: WebSocketEventMap[K]) => unknown,
    options?: boolean | AddEventListenerOptions,
  ): void;
  addEventListener(type: string, listener: EventListenerOrEventListenerObject | null, options?: boolean | AddEventListenerOptions): void;
  removeEventListener<K extends keyof WebSocketEventMap>(
    type: K,
    listener: (this: WebSocket, ev: WebSocketEventMap[K]) => unknown,
    options?: boolean | EventListenerOptions,
  ): void;
  removeEventListener(type: string, listener: EventListenerOrEventListenerObject | null, options?: boolean | EventListenerOptions): void;
}

export default WebSocket;

function closeEvent(code: number, reason: string, wasClean: boolean): CloseEvent {
  if (typeof CloseEvent !== "undefined") return new CloseEvent("close", { code, reason, wasClean });
  return Object.assign(new Event("close"), { code, reason, wasClean }) as CloseEvent;
}

/** Why the connection failed. */
function failure(err: unknown): string {
  if (isCallError(err) && err.message.includes("plugin is not used")) {
    return "the Go side does not use @mygo-plugins/websocket: call mygo.Use(websocket.Plugin) before App.Run";
  }
  return err instanceof Error ? err.message : String(err);
}

function toBase64(bytes: Uint8Array): string {
  const native = (bytes as { toBase64?: () => string }).toBase64;
  if (native) return native.call(bytes);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000) as unknown as number[]);
  }
  return btoa(s);
}

function fromBase64(s: string): Uint8Array {
  const native = (Uint8Array as { fromBase64?: (s: string) => Uint8Array }).fromBase64;
  if (native) return native(s);
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
