#!/usr/bin/env bash
# Runs the W0 Go and resource benchmarks twice and compares the A/A noise.
# Run from Voxt/: scripts/benchmarks.sh DIR [BENCHTIME]
set -euo pipefail

[ $# -ge 1 ] && [ $# -le 2 ] || { echo "usage: $0 DIR [BENCHTIME]" >&2; exit 2; }
cd "$(dirname "$0")/.."
mkdir -p "$1"
out="$(cd "$1" && pwd)"
benchtime="${2:-100x}"
stream_frames="$(python3 -c 'import json; print(len(json.load(open("testdata/w0-stream-lengths.json"))))')"
benchstat="${BENCHSTAT:-$(command -v benchstat || true)}"
[ -n "$benchstat" ] || { echo "install benchstat or set BENCHSTAT" >&2; exit 1; }
export CGO_ENABLED=0 GOMAXPROCS="${GOMAXPROCS:-4}"
rm -f "$out/a.txt" "$out/b.txt"
for iteration in {1..10}; do
	for run in a b; do
		go test -run '^$' -bench '^BenchmarkW0(AnswerCards|LongLists|MeetingDetailUpdates|MeetingOverlayHour|MeetingOverlayStates|OverlayStates|Pages|WaveformUpdate)$' -benchmem -count 1 -benchtime "$benchtime" . >>"$out/$run.txt"
		go test -run '^$' -bench '^BenchmarkW0AnswerStreaming$' -benchmem -count 1 -benchtime "${stream_frames}x" . >>"$out/$run.txt"
	done
done
"$benchstat" "$out/a.txt" "$out/b.txt" >"$out/benchstat.txt"
VOXTNEW_W0_RESOURCES=1 go test -run '^TestW0ResourceBaseline$' -count 10 -v . >"$out/resources.txt"
python3 -c 'import collections, pathlib, statistics, sys
samples = collections.defaultdict(lambda: collections.defaultdict(list))
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if "MEASURE resource" not in line:
        continue
    fields = line[line.index("MEASURE"):].split()
    phase = fields[2].split("=", 1)[1]
    for field in fields[3:]:
        key, value = field.split("=", 1)
        samples[phase][key].append(float(value))
lines = ["phase\tmetric\tmedian\tmin\tmax"]
for phase, metrics in samples.items():
    for metric, values in metrics.items():
        lines.append(f"{phase}\t{metric}\t{statistics.median(values):.3f}\t{min(values):.3f}\t{max(values):.3f}")
pathlib.Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
' "$out/resources.txt" "$out/resources-summary.txt"
python3 -c 'import pathlib, statistics, sys
runs = []
for name in sys.argv[1:3]:
    benchmarks = {}
    for line in pathlib.Path(name).read_text().splitlines():
        if not line.startswith("Benchmark"):
            continue
        fields = line.split()
        values = {}
        for index, field in enumerate(fields):
            if field in ("ns/op", "ns_per_update", "B/op", "allocs/op"):
                values[field] = float(fields[index - 1])
        benchmarks.setdefault(fields[0], []).append(values)
    runs.append(benchmarks)
lines = ["benchmark\ta_ns\tb_ns\tdelta_pct\ta_spread_pct\tb_spread_pct\ta_allocs\tb_allocs\ta_bytes\tb_bytes"]
for name in sorted(runs[0]):
    medians, spreads, allocs, sizes = [], [], [], []
    for measurements in runs:
        values = measurements[name]
        times = [value.get("ns_per_update", value["ns/op"]) for value in values]
        median = statistics.median(times)
        medians.append(median)
        spreads.append(100 * (max(times) - min(times)) / median)
        allocs.append(statistics.median(value["allocs/op"] for value in values))
        sizes.append(statistics.median(value["B/op"] for value in values))
    delta = 100 * (medians[1] - medians[0]) / medians[0]
    lines.append(f"{name}\t{medians[0]:.0f}\t{medians[1]:.0f}\t{delta:.2f}\t{spreads[0]:.2f}\t{spreads[1]:.2f}\t{allocs[0]:.0f}\t{allocs[1]:.0f}\t{sizes[0]:.0f}\t{sizes[1]:.0f}")
pathlib.Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
noisy = [line for line in lines[1:] if abs(float(line.split("\t")[3])) > 5]
if noisy:
    raise SystemExit("A/A median delta exceeds 5%: " + ", ".join(noisy))
unstable = [line for line in lines[1:] if max(float(line.split("\t")[4]), float(line.split("\t")[5])) > 5]
if unstable:
    raise SystemExit("A/A within-side spread exceeds 5%: " + ", ".join(unstable))
' "$out/a.txt" "$out/b.txt" "$out/summary.txt"
