// Runs MyGo's benchmarks and keeps their history, which the website's
// /benchmarks page shows. .github/workflows/bench.yml runs them on every
// push to main, on macOS, Linux and Windows:
//
//   bun scripts/bench.ts run [--e2e] [--count 6] [--benchtime 500ms] [--out file] [packages...]
//   bun scripts/bench.ts merge <data dir> <result files...>
//
// run measures the checkout it runs in: the Go benchmarks of every package
// that has some (or of the packages given), those of internal/e2e with
// --e2e, which need a desktop session, the size of release builds of two
// examples, and with --e2e their memory once idle (internal/idlemem).
// Packages run one at a time, each benchmark --count times, as each app,
// and the median of each metric is kept.
//
// merge adds results to a checkout of the benchmarks branch:
// history/<yyyy-mm-dd>.json keeps every commit of a day (UTC), small files
// that each push rewrites one of, and latest.json the last commits, by
// series, which the website fetches.
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";

const module = "github.com/egoist/mygo";

/** Values by unit, such as "ns/op", "B/op", "allocs/op" or "bytes". */
export type Metrics = Record<string, number>;

/** Metrics by package ("." for the root, "size" and "memory" for apps'), then benchmark. */
export type Results = Record<string, Record<string, Metrics>>;

/** The machine a run measured on. */
export interface Runner {
  arch: string;
  cpu: string;
  go: string;
}

/** What run measured: the results of one commit on one OS. */
export interface Run extends Runner {
  sha: string;
  date: string;
  message: string;
  os: string;
  count: number;
  results: Results;
}

/** A commit of history/<yyyy-mm-dd>.json, with the runs of each OS. */
export interface Commit {
  sha: string;
  date: string;
  message: string;
  runs: Record<string, Runner & { count: number; results: Results }>;
}

/** latest.json: the last commits, oldest first, and their values by series. */
export interface Latest {
  version: 1;
  commits: { sha: string; date: string; message: string }[];
  /** By OS: the runner of each commit, null where the OS has no run. */
  runners: Record<string, (Runner | null)[]>;
  /** By OS, package, benchmark and unit: the value at each commit, or null. */
  series: Record<string, Record<string, Record<string, Record<string, (number | null)[]>>>>;
}

/** How many commits latest.json keeps. */
export const latestCommits = 300;

/** The examples whose release builds are measured. */
const apps = ["hello", "counter-native"];

/** The apps with a webview, whose own process's memory counts apart from its processes' too. */
const webviewApps = new Set(["hello"]);

async function run(args: string[]): Promise<void> {
  let e2e = false;
  let count = 6;
  let benchtime = "500ms";
  let out = "";
  const pkgs: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--e2e") e2e = true;
    else if (arg === "--count") count = Number(args[++i]);
    else if (arg === "--benchtime") benchtime = args[++i]!;
    else if (arg === "--out") out = args[++i]!;
    else pkgs.push(arg);
  }
  if (!pkgs.length) pkgs.push(...(await benchmarkPackages(e2e)));

  const [goVersion, goos, goarch] = output(["go", "env", "GOVERSION", "GOOS", "GOARCH"]).split("\n");
  const [sha, date, message] = output(["git", "log", "-1", "--format=%H%n%cI%n%s"]).split("\n");
  // -p 1: packages run one at a time, or they would measure each other.
  const test = await goTest(
    ["-p", "1", "-run", "^$", "-bench", ".", "-benchmem", "-count", String(count), "-benchtime", benchtime, "-timeout", "60m", ...pkgs],
    e2e ? { MYGO_E2E: "1" } : {},
  );
  const { cpu, results } = parse(test.output);
  const measured = await measureApps(goos!, e2e, count);
  results.size = measured.size;
  if (measured.memory) results.memory = measured.memory;

  const result: Run = {
    sha: sha!,
    date: date!,
    message: message!,
    os: goos!,
    arch: goarch!,
    cpu,
    go: goVersion!,
    count,
    results,
  };
  if (out) await writeFile(out, JSON.stringify(result, null, 2) + "\n");
  print(results);
  // What did run is kept, and the failure shows.
  if (test.code !== 0) console.error(`go test exited with ${test.code}`);
  if (test.code !== 0 || measured.failed) process.exit(1);
}

/**
 * Returns the directories of the packages that have benchmarks on this
 * platform: those of internal/gpu/metal, say, only build on macOS.
 */
async function benchmarkPackages(e2e: boolean): Promise<string[]> {
  const dirs = new Set<string>();
  // Untracked files too, as the benchmarks being written.
  for (const file of output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*_test.go"]).split("\n")) {
    if (!file) continue;
    const dir = file.includes("/") ? file.slice(0, file.lastIndexOf("/")) : ".";
    if (!e2e && dir === "internal/e2e") continue;
    dirs.add(dir);
  }
  const pkgs = [...dirs].sort().map((dir) => (dir === "." ? "." : "./" + dir));
  // The test files that build here, by package; go list rather than ./...,
  // which skips testdata directories.
  const listed = output(["go", "list", "-e", "-f", '{{.Dir}}|{{join .TestGoFiles ","}},{{join .XTestGoFiles ","}}', ...pkgs]);
  const benchmarked: string[] = [];
  for (const line of listed.split("\n")) {
    const [dir, files] = line.split("|");
    for (const file of files!.split(",")) {
      if (file && /^func Benchmark/m.test(await readFile(join(dir!, file), "utf8"))) {
        const rel = relative(process.cwd(), dir!);
        benchmarked.push(rel ? "./" + rel : ".");
        break;
      }
    }
  }
  return benchmarked.sort();
}

/** Runs go test, showing its output as it comes. */
async function goTest(args: string[], env: Record<string, string>): Promise<{ output: string; code: number }> {
  console.log(`go test ${args.join(" ")}`);
  const proc = Bun.spawn(["go", "test", ...args], { stdout: "pipe", stderr: "inherit", env: { ...process.env, ...env } });
  const decoder = new TextDecoder();
  let text = "";
  for await (const chunk of proc.stdout) {
    const s = decoder.decode(chunk, { stream: true });
    text += s;
    process.stdout.write(s);
  }
  return { output: text, code: await proc.exited };
}

/**
 * Parses the output of go test -bench: the median of each metric over the
 * runs of each benchmark, named without "Benchmark" and GOMAXPROCS, and the
 * CPU go test reports. MB/s is left out: it follows from ns/op.
 */
export function parse(text: string): { cpu: string; results: Results } {
  const samples: Record<string, Record<string, Record<string, number[]>>> = {};
  let pkg = "";
  let cpu = "";
  let pending = "";
  for (const line of text.split(/\r?\n/)) {
    let m: RegExpExecArray | null;
    if ((m = /^pkg: (\S+)/.exec(line))) {
      pkg = m[1] === module ? "." : m[1]!.replace(module + "/", "");
      pending = "";
      continue;
    }
    if ((m = /^cpu: (.+)/.exec(line))) {
      cpu ||= m[1]!.trim();
      continue;
    }
    const fields = line.trim().split(/\s+/);
    let name = "";
    let values: string[];
    if (/^Benchmark\S/.test(fields[0]!)) {
      // A benchmark that logs has its name on a line of its own.
      if (fields.length === 1) {
        pending = fields[0]!;
        continue;
      }
      name = fields[0]!;
      values = fields.slice(1);
    } else if (pending && /^\d+$/.test(fields[0]!)) {
      name = pending;
      values = fields;
    } else continue;
    pending = "";
    // The iterations, then pairs of a value and its unit.
    if (!/^\d+$/.test(values[0]!) || values.length < 3) continue;
    name = name.replace(/^Benchmark/, "").replace(/-\d+$/, "");
    const metrics = ((samples[pkg] ??= {})[name] ??= {});
    for (let i = 1; i + 1 < values.length; i += 2) {
      const value = Number(values[i]);
      const unit = values[i + 1]!;
      if (unit === "MB/s" || !Number.isFinite(value)) continue;
      (metrics[unit] ??= []).push(value);
    }
  }
  const results: Results = {};
  for (const [pkg, benchmarks] of Object.entries(samples)) {
    for (const [name, units] of Object.entries(benchmarks)) {
      const metrics: Metrics = {};
      for (const [unit, values] of Object.entries(units)) metrics[unit] = round(median(values));
      (results[pkg] ??= {})[name] = metrics;
    }
  }
  return { cpu, results };
}

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** Rounds to 4 significant digits, more than the noise of any runner. */
function round(v: number): number {
  return Number(v.toPrecision(4));
}

/**
 * Builds the examples as mygo build does for a release, and returns their
 * sizes and, with e2e, their memory once idle, in "B": an app with its
 * processes, and a webview app's own process as "<app>/app". A failure
 * to measure memory leaves it out.
 */
async function measureApps(
  goos: string,
  e2e: boolean,
  count: number,
): Promise<{ size: Record<string, Metrics>; memory?: Record<string, Metrics>; failed?: boolean }> {
  const dir = await mkdtemp(join(tmpdir(), "mygo-bench-"));
  const size: Record<string, Metrics> = {};
  try {
    const bins: string[] = [];
    for (const app of apps) {
      if (!existsSync(join("examples", app))) continue;
      const bin = join(dir, goos === "windows" ? app + ".exe" : app);
      const ldflags = "-s -w -X github.com/egoist/mygo.production=1" + (goos === "windows" ? " -H=windowsgui" : "");
      output(["go", "build", "-trimpath", "-tags", "mygo_noinspector", "-ldflags", ldflags, "-o", bin, "./examples/" + app]);
      size[app] = { bytes: (await stat(bin)).size };
      bins.push(bin);
    }
    if (!e2e || !existsSync(join("internal", "idlemem"))) return { size };
    console.log(`go run ./internal/idlemem -count ${count}`);
    const proc = Bun.spawnSync(["go", "run", "./internal/idlemem", "-count", String(count), ...bins], { stderr: "inherit" });
    if (proc.exitCode !== 0) {
      console.error(`idlemem exited with ${proc.exitCode}`);
      return { size, failed: true };
    }
    const memory: Record<string, Metrics> = {};
    const idle: Record<string, { app: number; total: number }> = JSON.parse(proc.stdout.toString());
    for (const [app, m] of Object.entries(idle)) {
      memory[app] = { B: round(m.total) };
      if (webviewApps.has(app)) memory[`${app}/app`] = { B: round(m.app) };
    }
    return { size, memory };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

function output(cmd: string[]): string {
  const proc = Bun.spawnSync(cmd, { stderr: "inherit" });
  if (proc.exitCode !== 0) throw new Error(`${cmd.join(" ")} exited with ${proc.exitCode}`);
  return proc.stdout.toString().trim();
}

function print(results: Results): void {
  const rows = [["benchmark", "time/op", "B/op", "allocs/op"]];
  for (const [pkg, benchmarks] of Object.entries(results)) {
    for (const [name, m] of Object.entries(benchmarks)) {
      const bytes = m.bytes ?? m.B;
      const time = m["ns/op"] !== undefined ? formatNs(m["ns/op"]) : bytes !== undefined ? `${(bytes / 1e6).toFixed(2)} MB` : "";
      rows.push([`${pkg}/${name}`, time, String(m["B/op"] ?? ""), String(m["allocs/op"] ?? "")]);
    }
  }
  const widths = rows[0]!.map((_, i) => Math.max(...rows.map((r) => r[i]!.length)));
  for (const r of rows) console.log(r.map((c, i) => (i ? c.padStart(widths[i]!) : c.padEnd(widths[i]!))).join("  "));
}

function formatNs(ns: number): string {
  if (ns >= 1e9) return `${(ns / 1e9).toPrecision(3)} s`;
  if (ns >= 1e6) return `${(ns / 1e6).toPrecision(3)} ms`;
  if (ns >= 1e3) return `${(ns / 1e3).toPrecision(3)} µs`;
  return `${ns.toPrecision(3)} ns`;
}

/** Adds runs to the history in dir, and writes latest.json anew. */
export async function merge(dir: string, files: string[]): Promise<void> {
  const runs: Run[] = await Promise.all(files.map(async (f) => JSON.parse(await readFile(f, "utf8"))));
  const history = join(dir, "history");
  await mkdir(history, { recursive: true });
  const days = new Map<string, Commit[]>();
  for (const r of runs) {
    const day = new Date(r.date).toISOString().slice(0, 10);
    let commits = days.get(day);
    if (!commits) {
      commits = await readShard(join(history, day + ".json"));
      days.set(day, commits);
    }
    let commit = commits.find((c) => c.sha === r.sha);
    if (!commit) {
      commit = { sha: r.sha, date: r.date, message: r.message, runs: {} };
      commits.push(commit);
    }
    // A run measured again replaces the earlier one.
    commit.runs[r.os] = { arch: r.arch, cpu: r.cpu, go: r.go, count: r.count, results: r.results };
  }
  for (const [day, commits] of days) {
    commits.sort((a, b) => Date.parse(a.date) - Date.parse(b.date));
    // A commit a line, for readable diffs.
    await writeFile(join(history, day + ".json"), "[\n" + commits.map((c) => JSON.stringify(c)).join(",\n") + "\n]\n");
  }

  const recent: Commit[] = [];
  const shards = (await readdir(history)).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)).sort().reverse();
  for (const shard of shards) {
    recent.unshift(...(await readShard(join(history, shard))));
    if (recent.length >= latestCommits) break;
  }
  await writeFile(join(dir, "latest.json"), JSON.stringify(latest(recent.slice(-latestCommits))));
  await writeFile(join(dir, "README.md"), readme);
}

async function readShard(file: string): Promise<Commit[]> {
  return existsSync(file) ? JSON.parse(await readFile(file, "utf8")) : [];
}

/** Turns commits, oldest first, into the series of latest.json. */
export function latest(commits: Commit[]): Latest {
  const data: Latest = {
    version: 1,
    commits: commits.map(({ sha, date, message }) => ({ sha, date, message })),
    runners: {},
    series: {},
  };
  const n = commits.length;
  commits.forEach((c, i) => {
    for (const [os, r] of Object.entries(c.runs)) {
      (data.runners[os] ??= Array(n).fill(null))[i] = { arch: r.arch, cpu: r.cpu, go: r.go };
      const byPkg = (data.series[os] ??= {});
      for (const [pkg, benchmarks] of Object.entries(r.results)) {
        for (const [name, metrics] of Object.entries(benchmarks)) {
          const byUnit = ((byPkg[pkg] ??= {})[name] ??= {});
          for (const [unit, value] of Object.entries(metrics)) (byUnit[unit] ??= Array(n).fill(null))[i] = value;
        }
      }
    }
  });
  return data;
}

const readme = `# MyGo's benchmarks

The results of MyGo's benchmarks, which .github/workflows/bench.yml of the
main branch measures on every push, written by scripts/bench.ts and shown at
https://mygo.egoist.dev/benchmarks.

- history/<yyyy-mm-dd>.json: every commit measured that day (UTC), with each OS's runner and results.
- latest.json: the last ${latestCommits} commits, by series, which the website fetches.

Its commits skip CI ([skip ci]), the website's builds included.
`;

// Last, once the module's constants are set.
if (import.meta.main) {
  const [command, ...args] = process.argv.slice(2);
  if (command === "run") await run(args);
  else if (command === "merge" && args.length >= 2) await merge(args[0]!, args.slice(1));
  else {
    console.error("usage: bun scripts/bench.ts run [--e2e] [--count n] [--benchtime d] [--out file] [packages...]");
    console.error("       bun scripts/bench.ts merge <data dir> <result files...>");
    process.exit(2);
  }
}
