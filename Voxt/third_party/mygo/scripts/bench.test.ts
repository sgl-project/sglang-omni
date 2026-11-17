import { describe, expect, test } from "bun:test";
import { mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Latest, type Run, latestCommits, merge, parse } from "./bench.ts";

const output = `goos: darwin
goarch: arm64
pkg: github.com/egoist/mygo
cpu: Apple M5
BenchmarkCallNoop-10       	   43904	     27364 ns/op	   2.89 MB/s	     719 B/op	      14 allocs/op
BenchmarkCallNoop-10       	   43000	     27000 ns/op	   2.91 MB/s	     720 B/op	      14 allocs/op
BenchmarkCallNoop-10       	   44000	     29000 ns/op	   2.80 MB/s	     721 B/op	      14 allocs/op
PASS
ok  	github.com/egoist/mygo	9.634s
skipping e2e tests; set MYGO_E2E=1 to run them in a desktop session
ok  	github.com/egoist/mygo/internal/e2e	0.376s
goos: darwin
goarch: arm64
pkg: github.com/egoist/mygo/ui
cpu: Apple M5
BenchmarkTextAreaType/1000-10        	    3018	   1258632 ns/op	  199799 B/op	     248 allocs/op
BenchmarkTextAreaType/1000-10        	    3018	   1258000 ns/op	  199801 B/op	     250 allocs/op
BenchmarkLogs-10
2026/10/05 12:00:00 a log line
     100	       0.5 ns/op
--- SKIP: BenchmarkOutput
    bench_test.go:15: no library
BenchmarkDiff-10    	       1	 123 patch-bytes	 456 file-bytes
PASS
`;

describe("parse", () => {
  const { cpu, results } = parse(output);

  test("takes the median of each metric, without MB/s", () => {
    expect(cpu).toBe("Apple M5");
    expect(results["."]).toEqual({ CallNoop: { "ns/op": 27360, "B/op": 720, "allocs/op": 14 } });
  });

  test("names benchmarks without Benchmark and GOMAXPROCS, sub-benchmarks included", () => {
    expect(results.ui?.["TextAreaType/1000"]).toEqual({ "ns/op": 1258000, "B/op": 199800, "allocs/op": 249 });
  });

  test("reads results printed after a benchmark's logs", () => {
    expect(results.ui?.Logs).toEqual({ "ns/op": 0.5 });
  });

  test("keeps custom metrics", () => {
    expect(results.ui?.Diff).toEqual({ "patch-bytes": 123, "file-bytes": 456 });
  });

  test("leaves out skipped benchmarks", () => {
    expect(results.ui?.Output).toBeUndefined();
  });
});

function run(sha: string, date: string, os: string, ns: number): Run {
  return {
    sha,
    date,
    message: `commit ${sha}`,
    os,
    arch: "arm64",
    cpu: "Apple M5",
    go: "go1.27.1",
    count: 6,
    results: { ".": { CallNoop: { "ns/op": ns, "B/op": 720 } }, size: { hello: { bytes: 8e6 } } },
  };
}

async function mergeRuns(dir: string, runs: Run[]): Promise<void> {
  const files = await Promise.all(
    runs.map(async (r, i) => {
      const file = join(dir, `run-${i}.json`);
      await writeFile(file, JSON.stringify(r));
      return file;
    }),
  );
  await merge(join(dir, "data"), files);
}

async function readLatest(dir: string): Promise<Latest> {
  return JSON.parse(await readFile(join(dir, "data", "latest.json"), "utf8"));
}

describe("merge", () => {
  test("keeps each commit's runs by OS and day, and their series", async () => {
    const dir = await mkdtemp(join(tmpdir(), "bench-test-"));
    await mergeRuns(dir, [run("b", "2026-10-02T00:00:00Z", "linux", 200), run("b", "2026-10-02T00:00:00Z", "darwin", 100)]);
    // An older commit measured later, another OS's run of a commit, and a
    // run measured again.
    await mergeRuns(dir, [
      run("a", "2026-09-30T23:00:00-02:00", "darwin", 90),
      run("c", "2026-10-03T00:00:00Z", "darwin", 110),
      run("b", "2026-10-02T00:00:00Z", "darwin", 105),
    ]);
    expect((await readdir(join(dir, "data", "history"))).sort()).toEqual(["2026-10-01.json", "2026-10-02.json", "2026-10-03.json"]);
    const data = await readLatest(dir);
    expect(data.commits.map((c) => c.sha)).toEqual(["a", "b", "c"]);
    expect(data.series.darwin?.["."]?.CallNoop?.["ns/op"]).toEqual([90, 105, 110]);
    expect(data.series.linux?.["."]?.CallNoop?.["ns/op"]).toEqual([null, 200, null]);
    expect(data.series.darwin?.size?.hello?.bytes).toEqual([8e6, 8e6, 8e6]);
    expect(data.runners.linux).toEqual([null, { arch: "arm64", cpu: "Apple M5", go: "go1.27.1" }, null]);
  });

  test("keeps the last commits in latest.json, and every day in history", async () => {
    const dir = await mkdtemp(join(tmpdir(), "bench-test-"));
    const runs = Array.from({ length: latestCommits + 5 }, (_, i) =>
      run(`c${i}`, new Date(Date.UTC(2026, 8, 1) + i * 3_600_000 * 4).toISOString(), "linux", i),
    );
    await mergeRuns(dir, runs);
    // Six commits a day from September 1st.
    expect((await readdir(join(dir, "data", "history"))).length).toBe(Math.ceil((latestCommits + 5) / 6));
    const data = await readLatest(dir);
    expect(data.commits.length).toBe(latestCommits);
    expect(data.commits[0]?.sha).toBe("c5");
    expect(data.series.linux?.["."]?.CallNoop?.["ns/op"]?.at(-1)).toBe(latestCommits + 4);
  });
});
