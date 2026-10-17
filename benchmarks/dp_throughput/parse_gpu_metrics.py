# SPDX-License-Identifier: Apache-2.0
"""Parse an nsys GPU-metrics report -> averaged device-level SOL throughput %.

Usage: python parse_gpu_metrics.py <report.nsys-rep> [label]

Emits SMs Active %, Tensor Active %, DRAM Read/Write BW %, Compute Warps %,
averaged over the capture window. These are NVIDIA's device-level Speed-Of-Light
throughput counters (sampled by `nsys --gpu-metrics`), so they measure the WHOLE
physical GPU across all processes on it -- which is why they are the right
cross-process metric when several replicas share one card (per-process CUPTI
timelines cannot see the other replica). GPU idle fraction ~= 100 - SMs Active %.
"""
import os
import sqlite3
import subprocess
import sys

rep = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(rep)
sqlite_path = rep.replace(".nsys-rep", ".sqlite")

subprocess.run(
    [
        "nsys",
        "export",
        "--type",
        "sqlite",
        "--force-overwrite",
        "true",
        "-o",
        sqlite_path,
        rep,
    ],
    capture_output=True,
    text=True,
)

c = sqlite3.connect(sqlite_path)
names = {r[3]: r[4] for r in c.execute("select * from TARGET_INFO_GPU_METRICS")}
avg = {
    mid: av
    for mid, av in c.execute(
        "select metricId, avg(value) from GPU_METRICS group by metricId"
    )
}
n_samples = c.execute("select count(distinct timestamp) from GPU_METRICS").fetchone()[0]


def g(substr):
    for mid, nm in names.items():
        if substr.lower() in nm.lower():
            return avg.get(mid, float("nan")), nm
    return float("nan"), "?"


sm, _ = g("SMs Active")
tensor, _ = g("Tensor Active")
dr, _ = g("DRAM Read")
dw, _ = g("DRAM Write")
cw, _ = g("Compute Warps in Flight [Throughput")

print(f"==== GPU-METRICS SOL [{label}] (device-level, {n_samples} samples) ====")
print(
    f"  SMs Active    [Throughput %] : {sm:6.1f}   (compute occupancy; GPU idle ~= 100 - this)"
)
print(f"  Tensor Active [Throughput %] : {tensor:6.1f}   (tensor-core utilization)")
print(f"  Compute Warps [Throughput %] : {cw:6.1f}")
print(f"  DRAM Read BW  [Throughput %] : {dr:6.1f}   (dram__throughput proxy)")
print(f"  DRAM Write BW [Throughput %] : {dw:6.1f}")
print(f"  DRAM R+W total approx        : {dr+dw:6.1f}")
hi_compute = max(sm, tensor)
hi_mem = dr + dw
if hi_compute < 25 and hi_mem < 25:
    hint = "GPU UNDER-UTILIZED on both axes -> HOST/LAUNCH-bound"
elif hi_compute >= 2.0 * max(hi_mem, 1):
    hint = "compute axis dominates -> COMPUTE-leaning"
elif hi_mem >= 1.3 * max(hi_compute, 1):
    hint = "memory axis dominates -> MEMORY-BW-leaning"
else:
    hint = "mixed / inconclusive"
print(f"  >>> device-level hint: {hint}")
