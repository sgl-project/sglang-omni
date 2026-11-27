# SPDX-License-Identifier: Apache-2.0
"""Prove the server was saturated during the measurement window (client-control check).

Reads window_start.txt / window_end.txt from <rundir> (written by run_condition.sh),
scans each server log for `#running-req: N, #queue-req: M` lines timestamped inside the
window, and reports median/min/max. running-req near the engine cap with queue-req > 0
means the server always had work, so any GPU idle is server-internal, not client
starvation. min running-req dropping toward 0 means the client failed to keep the server
fed (a client artifact) -- fix the load generator, not the server.

Usage: python analyze_saturation.py <rundir> <server_log> [<server_log> ...]
"""
import datetime
import re
import statistics as st
import sys

rundir = sys.argv[1]
logs = sys.argv[2:]
ws = float(open(f"{rundir}/window_start.txt").read().strip())
we = float(open(f"{rundir}/window_end.txt").read().strip())

pat = re.compile(r"running-req: (\d+), #queue-req: (\d+)")
tpat = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
for log in logs:
    run, q = [], []
    try:
        for line in open(log, errors="ignore"):
            m = pat.search(line)
            tm = tpat.match(line)
            if m and tm:
                t = datetime.datetime.strptime(
                    tm.group(1), "%Y-%m-%d %H:%M:%S"
                ).timestamp()
                if ws <= t <= we:
                    run.append(int(m.group(1)))
                    q.append(int(m.group(2)))
    except FileNotFoundError:
        print(f"  {log}: NOT FOUND")
        continue
    if run:
        print(
            f"  {log}: samples={len(run)} running-req med={st.median(run):.0f} "
            f"min={min(run)} max={max(run)} | queue med={st.median(q):.0f} "
            f"min={min(q)} max={max(q)}"
        )
    else:
        print(f"  {log}: no in-window running-req lines (window {int(we-ws)}s)")
