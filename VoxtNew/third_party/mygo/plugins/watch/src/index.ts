/**
 * The watch plugin of MyGo: changes to the files and directories the app
 * lets pages watch, by the names it gives them. The Go side must use the
 * plugin with those directories:
 *
 * ```go
 * mygo.Use(watch.New(watch.Options{ // github.com/egoist/mygo/plugins/watch
 * 	Roots: map[string]watch.Root{"project": {Path: dir, Recursive: true}},
 * }))
 * ```
 *
 * ```ts
 * import { watch } from "@mygo-plugins/watch";
 *
 * const watcher = await watch("project", { recursive: true });
 * for await (const events of watcher) {
 *   for (const event of events) console.log(event.op, event.path);
 * }
 * ```
 *
 * @module
 */
import { Channel, call } from "mygo-runtime";

/** The kind of a {@link FileEvent}. */
export type FileOp = "create" | "write" | "remove" | "rename";

/**
 * A change in a watched directory or file. Events say what is different
 * once changes stop, not every step that made it so: treat them as hints
 * to read the files again.
 */
export interface FileEvent {
  op: FileOp;
  /** The path relative to what is watched, separated by slashes; "." is what is watched itself. */
  path: string;
  /** Where a rename moved from. */
  oldPath?: string;
  isDir: boolean;
}

/** Options of {@link watch}. */
export interface WatchOptions {
  /** What to watch in the root, a directory or a file, relative to it and separated by slashes: the root itself by default. */
  path?: string;
  /** Watch the directories in the directory too, if the app allows it. */
  recursive?: boolean;
  /** How long changes must stop before they are reported: 50 ms when 0 or omitted, right away when negative. */
  debounceMs?: number;
  /** Stops the watch when aborted. */
  signal?: AbortSignal;
}

/** Why a watch failed or stopped; see {@link WatchError}. */
export type WatchErrorCode = "denied" | "limit" | "not-found" | "unsupported" | "path-encoding" | "io";

/** A watch that failed to start or stopped, with the reason as its code. */
export class WatchError extends Error {
  constructor(
    readonly code: WatchErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "WatchError";
  }
}

/**
 * A running watch: iterate it for the changes, a batch of events each time
 * they stop. Go doesn't look again until the page has taken a batch, so
 * nothing is lost while the page is busy.
 */
export interface Watcher extends AsyncIterable<FileEvent[]> {
  /** Settles once Go has stopped the watch, rejecting with the error that stopped it. */
  readonly closed: Promise<void>;
  /** Stops the watch; it resolves once Go has. */
  close(): Promise<void>;
}

/** What Go streams. */
interface Part {
  type: "ready" | "events";
  events?: FileEvent[];
}

/** How the watch ended, the result of Go's call: the code of what failed, if anything. */
interface Outcome {
  code?: string;
  message?: string;
}

const codes: readonly string[] = ["denied", "limit", "not-found", "unsupported", "path-encoding", "io"];

/**
 * Watches the directory or file the app named `root`, or `path` in it, and
 * resolves once everything is watched. What is already there is not
 * reported. When it is removed, renamed or replaced, the last batch has a
 * "remove" of ".", and the iteration ends.
 */
export async function watch(root: string, options: WatchOptions = {}): Promise<Watcher> {
  const debounceMs = options.debounceMs ?? 0;
  if (!Number.isInteger(debounceMs) || debounceMs > 60000) {
    throw new TypeError("debounceMs must be an integer up to 60000");
  }
  const path = options.path ?? ".";
  if (typeof path !== "string") throw new TypeError("path must be a string");
  if (options.signal?.aborted) throw options.signal.reason;
  const parts = new Channel<Part>();
  const stream = parts[Symbol.asyncIterator]();
  // The first error wins: the signal's reason, or the call's.
  let terminal: { error: unknown } | undefined;
  let closing = false;
  let finished = false;
  let iterating = false;
  const record = (error: unknown) => {
    terminal ??= { error };
  };
  const abort = () => {
    if (finished || closing) return;
    record(options.signal!.reason);
    closing = true;
    parts.close();
  };
  options.signal?.addEventListener("abort", abort, { once: true });
  const closed = call<Outcome | null>("plugin:watch.Watch", { root, path, recursive: options.recursive ?? false, debounceMs }, parts)
    .then(
      (outcome) => {
        if (outcome?.code) {
          record(
            codes.includes(outcome.code)
              ? new WatchError(outcome.code as WatchErrorCode, outcome.message ?? "filesystem watch failed")
              : new Error("Invalid watch outcome"),
          );
        }
        if (terminal) throw terminal.error;
      },
      (err: unknown) => {
        record(err);
        throw terminal!.error;
      },
    )
    .finally(() => {
      finished = true;
      options.signal?.removeEventListener("abort", abort);
    });
  void closed.catch(() => {});
  const close = async () => {
    if (!closing && !finished) {
      closing = true;
      parts.close();
    }
    await closed.catch(() => {});
  };
  try {
    const first = await stream.next();
    if (first.done) {
      await closed;
      throw new Error("Watch ended before ready");
    }
    if (first.value.type !== "ready") {
      parts.close();
      await closed;
      throw new Error("Invalid watch ready response");
    }
    if (terminal) {
      await closed;
      throw terminal.error;
    }
  } catch (err) {
    parts.close();
    options.signal?.removeEventListener("abort", abort);
    throw err;
  }
  return {
    closed,
    close,
    [Symbol.asyncIterator]() {
      if (iterating) throw new Error("A watch supports one iterator");
      iterating = true;
      return (async function* () {
        try {
          while (!closing) {
            const next = await stream.next();
            if (next.done) break;
            if (next.value.events?.length && !closing) yield next.value.events;
          }
          await closed;
        } finally {
          await close();
          iterating = false;
        }
      })();
    },
  };
}
