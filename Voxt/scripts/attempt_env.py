#!/usr/bin/env python3
"""Write one attempt's environment record (mac-latency.sh, per attempt).

usage: attempt_env.py OUT BEFORE_OTHERS BEFORE_POWER BEFORE_PRESSURE AFTER_OTHERS AFTER_PRESSURE

Contenders and AC state are checked before and after the attempt's go-test
process, so a contender or a pressure spike that lives only inside it
still rejects the attempt. Pressure is kern.memorystatus_vm_pressure_level
(1 is normal).
"""
import json
import sys


def reasons(others, power):
    out = []
    if others:
        out.append("contender: " + others.splitlines()[0][:120])
    if "AC Power" not in power:
        out.append("not on AC power")
    return out


def main(argv):
    if len(argv) != 7:
        print("usage: attempt_env.py OUT BEFORE_OTHERS BEFORE_POWER BEFORE_PRESSURE AFTER_OTHERS AFTER_PRESSURE", file=sys.stderr)
        return 2
    out_path, before_o, before_p, before_pr, after_o, after_pr = argv[1:7]
    env = {
        "before": {"others": before_o, "pressure": before_pr},
        "after": {"others": after_o, "pressure": after_pr},
        "environment_reasons": reasons(before_o, before_p) + reasons(after_o, before_p),
        "peak_pressure": "normal" if before_pr == "1" and after_pr == "1" else "warn",
    }
    with open(out_path, "w") as fh:
        json.dump(env, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
