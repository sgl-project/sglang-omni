#!/usr/bin/env node
// Runs the mygo binary of this platform with the arguments of this script.
import { spawn } from "node:child_process";
import { binaryPath } from "../index.js";

let bin;
try {
  bin = binaryPath();
} catch (err) {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
}
const args = process.argv.slice(2);

if (process.platform !== "win32" && typeof process.execve === "function") {
  // Become the binary (Node.js 23.11 and later, Bun), so that signals and
  // the exit status are its own.
  process.execve(bin, [bin, ...args]);
}

const child = spawn(bin, args, { stdio: "inherit" });
// Ctrl+C reaches the binary from the terminal too: let it decide when to
// exit, which mygo dev takes a moment for.
process.on("SIGINT", () => {});
process.on("SIGTERM", () => child.kill("SIGTERM"));
child.on("error", (err) => {
  console.error(`mygo-cli: cannot run ${bin}: ${err.message}`);
  process.exit(1);
});
child.on("exit", (code, signal) => {
  if (signal) {
    process.removeAllListeners(signal);
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 1);
  }
});
