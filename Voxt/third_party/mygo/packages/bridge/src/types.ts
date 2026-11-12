// The public types are those of the mygo-runtime package.
import type { Direction, Platform } from "../../runtime/src/types";

export type {
  Channel,
  Direction,
  LocaleChange,
  Platform,
  Runtime,
  ThemePreferences,
  WindowControls,
} from "../../runtime/src/types";

/** Configuration injected by the Go side in front of the bridge. */
export interface BridgeConfig {
  platform: Platform;
  windowId: number;
  version: string;
  /** Prefixes every message, so Go can tell ours from other frames'. */
  secret: string;
  /** The language the app shows, e.g. "zh-Hans", and its direction. */
  locale: string;
  dir: Direction;
}

/** Messages posted from the page to Go. */
export type Outgoing =
  | { t: "call"; id: number; k: string; m: string; a: unknown[] }
  /** Channel c took the values up to the n-th. */
  | { t: "chan-ack"; c: number; k: string; n: number }
  | { t: "chan-close"; c: number; k: string }
  | { t: "dom-ready" }
  | { t: "drag" }
  | { t: "dblclick" };

/** Messages delivered from Go to the page. */
export type Incoming =
  | { t: "event"; n: string; p?: unknown }
  | { t: "reply"; id: number; k: string; ok: boolean; v?: unknown; e?: string }
  /** A value p of channel c, which asks for an acknowledgment with a, or its end. */
  | { t: "chan"; c: number; k: string; p?: unknown; a?: 1; end?: true };
