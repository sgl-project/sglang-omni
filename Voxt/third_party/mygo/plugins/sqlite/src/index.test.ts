import { afterEach, expect, test } from "bun:test";
import { fakeGo } from "../../fake-go";
import { open } from "./index";

let go: ReturnType<typeof fakeGo> | undefined;
afterEach(() => go?.uninstall());

test("encodes parameters and decodes every SQLite storage class", async () => {
  go = fakeGo({
    "plugin:sqlite.Open": () => "db-1",
    "plugin:sqlite.Query": () => ({
      columns: ["n", "n", "big", "real", "text", "blob", "emptyBlob", "emptyText", "zero", "negative"],
      rows: [[{ type: "null" }, { type: "integer", integer: "42" }, { type: "integer", integer: "9223372036854775807" },
        { type: "real", real: 0.25 }, { type: "text", text: "中文\0🙂" }, { type: "blob", blob: "AP8=" },
        { type: "blob" }, { type: "text" }, { type: "real" }, { type: "integer", integer: "-9223372036854775808" }]],
    }),
  });
  const db = await open("app.db", { readOnly: true });
  const blob = new Uint8Array([9, 0, 255, 9]).subarray(1, 3);
  const rows = await db.query("SELECT ?, ?, ?, ?, ?, ?, ?, ?", [null, true, 42, -9223372036854775808n, 0.25, "中文\0🙂", blob, new Uint8Array()]);
  expect(rows.columns.slice(0, 2)).toEqual(["n", "n"]);
  expect(rows.rows[0]).toEqual([null, 42, 9223372036854775807n, 0.25, "中文\0🙂", new Uint8Array([0, 255]), new Uint8Array(), "", 0, -9223372036854775808n]);
  expect(go.calls[0]).toEqual(["plugin:sqlite.Open", ["app.db", { readOnly: true }]]);
  expect(go.calls[1]![1][2]).toEqual([
    { type: "null" }, { type: "integer", integer: "1" }, { type: "integer", integer: "42" },
    { type: "integer", integer: "-9223372036854775808" }, { type: "real", real: 0.25 },
    { type: "text", text: "中文\0🙂" }, { type: "blob", blob: "AP8=" }, { type: "blob", blob: "" },
  ]);
});

test("execute and transaction preserve rowids, and send a single atomic batch", async () => {
  const reply = { changes: 1, lastInsertId: "9007199254740993" };
  go = fakeGo({ "plugin:sqlite.Open": () => "db", "plugin:sqlite.Execute": () => reply, "plugin:sqlite.Transaction": () => [reply, reply] });
  const db = await open(":memory:");
  expect(await db.execute("INSERT", [false])).toEqual({ changes: 1, lastInsertId: 9007199254740993n });
  expect(await db.transaction([{ sql: "INSERT", params: [1] }, { sql: "UPDATE" }])).toEqual([
    { changes: 1, lastInsertId: 9007199254740993n }, { changes: 1, lastInsertId: 9007199254740993n },
  ]);
  expect(go.calls[2]).toEqual(["plugin:sqlite.Transaction", ["db", [{ sql: "INSERT", args: [{ type: "integer", integer: "1" }] }, { sql: "UPDATE", args: [] }]]]);
});

test("rejects unsafe parameters before making an IPC call", async () => {
  go = fakeGo({ "plugin:sqlite.Open": () => "db" });
  const db = await open(":memory:");
  for (const param of [NaN, Infinity, -Infinity, Number.MAX_SAFE_INTEGER + 1, 1n << 63n, -(1n << 63n) - 1n]) {
    await expect(db.execute("SELECT ?", [param])).rejects.toThrow("sqlite:");
  }
  await expect(db.transaction([{ sql: "SELECT 1" }, { sql: "SELECT ?", params: [NaN] }])).rejects.toThrow("finite");
  expect(go.calls.length).toBe(1);
});

test("large BLOBs do not overflow the browser's argument stack", async () => {
  go = fakeGo({ "plugin:sqlite.Open": () => "db", "plugin:sqlite.Execute": () => ({ changes: 0, lastInsertId: "0" }) });
  const db = await open(":memory:");
  const bytes = new Uint8Array(256 * 1024).fill(255);
  await db.execute("SELECT ?", [bytes]);
  const [arg] = go.calls[1]![1][2] as [{ blob: string }];
  expect(atob(arg.blob).length).toBe(bytes.length);
});

test("close is idempotent and rejects further use", async () => {
  go = fakeGo({ "plugin:sqlite.Open": () => "db", "plugin:sqlite.Close": () => undefined });
  const db = await open(":memory:");
  const a = db.close();
  const b = db.close();
  expect(a).toBe(b);
  await expect(db.query("SELECT 1")).rejects.toThrow("closed");
  await Promise.all([a, b]);
  await db.close();
  await expect(db.execute("SELECT 1")).rejects.toThrow("closed");
  expect(go.calls.length).toBe(2);
});

test("propagates Go errors and permits retrying a failed close", async () => {
  let attempts = 0;
  go = fakeGo({
    "plugin:sqlite.Open": () => "db",
    "plugin:sqlite.Query": () => { throw new Error("sqlite: no such table"); },
    "plugin:sqlite.Close": () => { if (++attempts === 1) throw new Error("close failed"); },
  });
  const db = await open(":memory:");
  await expect(db.query("SELECT * FROM missing")).rejects.toThrow("no such table");
  await expect(db.close()).rejects.toThrow("close failed");
  await db.close();
  expect(attempts).toBe(2);
});

test("decodes infinite REAL results without losing their SQLite type", async () => {
  go = fakeGo({
    "plugin:sqlite.Open": () => "db",
    "plugin:sqlite.Query": () => ({ columns: ["positive", "negative"], rows: [[{ type: "real", special: "Infinity" }, { type: "real", special: "-Infinity" }]] }),
  });
  const db = await open(":memory:");
  expect((await db.query("SELECT 1e999, -1e999")).rows).toEqual([[Infinity, -Infinity]]);
});
