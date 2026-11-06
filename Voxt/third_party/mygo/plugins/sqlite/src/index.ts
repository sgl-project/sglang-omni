import { call } from "mygo-runtime";

/** SQLite values: integers beyond Number's safe range are returned as bigint. */
export type Value = null | number | bigint | string | Uint8Array;
/** Booleans bind as SQLite integers 0 and 1. */
export type Parameter = Value | boolean;
export interface OpenOptions { readOnly?: boolean }
export interface Result {
  changes: number;
  /** The last inserted rowid on this connection, without precision loss. */
  lastInsertId: bigint;
}
/** Ordered columns and cells preserve duplicate column names. */
export interface Rows { columns: string[]; rows: Value[][] }
export interface Statement { sql: string; params?: readonly Parameter[] }

interface WireValue { type: string; integer?: string; real?: number; special?: "Infinity" | "-Infinity" | "NaN"; text?: string; blob?: string }
interface WireResult { changes: number; lastInsertId: string }
interface WireRows { columns: string[]; rows: WireValue[][] }

/** Opens a file name in the Go plugin's directory, or a private :memory: database. */
export async function open(name: string, options: OpenOptions = {}): Promise<Database> {
  return Database.open(name, options);
}

/** A page-owned connection. It closes when the page navigates or the app quits. */
export class Database {
  private closing?: Promise<void>;
  private closed = false;
  private constructor(private readonly id: string) {}

  static async open(name: string, options: OpenOptions = {}): Promise<Database> {
    const id = await call<string>("plugin:sqlite.Open", name, { readOnly: options.readOnly ?? false });
    return new Database(id);
  }

  /** Executes exactly one statement. Use placeholders, never interpolate values. */
  async execute(sql: string, params: readonly Parameter[] = []): Promise<Result> {
    this.assertOpen();
    return result(await call<WireResult>("plugin:sqlite.Execute", this.id, sql, params.map(encode)));
  }

  /** Queries one statement, including RETURNING; select a bounded page for large results. */
  async query(sql: string, params: readonly Parameter[] = []): Promise<Rows> {
    this.assertOpen();
    const value = await call<WireRows>("plugin:sqlite.Query", this.id, sql, params.map(encode));
    return { columns: value.columns, rows: value.rows.map((row) => row.map(decode)) };
  }

  /** Runs an atomic BEGIN IMMEDIATE batch. A failure rolls back every statement. */
  async transaction(statements: readonly Statement[]): Promise<Result[]> {
    this.assertOpen();
    const batch = statements.map((stmt) => ({ sql: stmt.sql, args: (stmt.params ?? []).map(encode) }));
    return (await call<WireResult[]>("plugin:sqlite.Transaction", this.id, batch)).map(result);
  }

  /** Closes this connection. Concurrent and repeated closes share the same call. */
  close(): Promise<void> {
    if (this.closed) return Promise.resolve();
    if (!this.closing) {
      this.closing = call<void>("plugin:sqlite.Close", this.id).then(
        () => { this.closed = true; },
        (error: unknown) => { this.closing = undefined; throw error; },
      );
    }
    return this.closing;
  }

  private assertOpen(): void {
    if (this.closed || this.closing) throw new Error("sqlite: database is closed");
  }
}

function result(value: WireResult): Result {
  return { changes: value.changes, lastInsertId: BigInt(value.lastInsertId) };
}

function encode(value: Parameter): WireValue {
  if (value === null) return { type: "null" };
  if (typeof value === "boolean") return { type: "integer", integer: value ? "1" : "0" };
  if (typeof value === "bigint") {
    if (value < -(1n << 63n) || value >= (1n << 63n)) throw new RangeError("sqlite: integer exceeds signed 64-bit range");
    return { type: "integer", integer: String(value) };
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("sqlite: number must be finite");
    if (Number.isInteger(value)) {
      if (!Number.isSafeInteger(value)) throw new RangeError("sqlite: use bigint for integers outside Number's safe range");
      return { type: "integer", integer: String(value) };
    }
    return { type: "real", real: value };
  }
  if (typeof value === "string") return { type: "text", text: value };
  if (value instanceof Uint8Array) {
    let text = "";
    for (let i = 0; i < value.length; i += 8192) text += String.fromCharCode(...value.subarray(i, i + 8192));
    return { type: "blob", blob: btoa(text) };
  }
  throw new TypeError("sqlite: unsupported parameter");
}

function decode(value: WireValue): Value {
  switch (value.type) {
    case "null": return null;
    case "integer": {
      const n = BigInt(value.integer!);
      return n >= BigInt(Number.MIN_SAFE_INTEGER) && n <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(n) : n;
    }
    case "real": return Number(value.special ?? value.real ?? 0);
    case "text": return value.text ?? "";
    case "blob": return Uint8Array.from(atob(value.blob ?? ""), (c) => c.charCodeAt(0));
    default: throw new TypeError(`sqlite: unknown value type ${value.type}`);
  }
}
