# SPDX-License-Identifier: Apache-2.0
"""Write a BVH file's HIERARCHY as the skeleton JSON an avatar_worker bvh_motion stage publishes in its port: joint names, parents (-1 for the root), rest offsets, each joint's rotation channel order, and optional face channel names.

python examples/avatar_workers/bvh_skeleton.py template.bvh skeleton.json [--face-names names.txt] [--up y]

The up axis is the one the rows' root positions and poses use, not the template file's: SentiAvatar's rows are Y up (the release swaps the root to (x, z, y) before writing a .bvh).
"""

from __future__ import annotations

import argparse
import json


def parse_hierarchy(text: str) -> dict:
    joints, parents, offsets, orders = [], [], [], []
    stack: list[int] = []
    pending_parent = -1
    is_end_site = False
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        head = words[0]
        if head == "MOTION":
            break
        elif head in ("ROOT", "JOINT"):
            joints.append(words[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            orders.append("")
            pending_parent = len(joints) - 1
        elif head == "End":
            is_end_site = True
        elif head == "{":
            stack.append(-2 if is_end_site else pending_parent)
        elif head == "}":
            popped = stack.pop()
            is_end_site = False if popped == -2 else is_end_site
        elif head == "OFFSET" and not is_end_site:
            offsets[stack[-1]] = [float(value) for value in words[1:4]]
        elif head == "CHANNELS":
            orders[stack[-1]] = "".join(
                word[0].lower() for word in words[2:] if word.endswith("rotation")
            )
    return {
        "joints": joints,
        "parents": parents,
        "offsets": offsets,
        "rotation_orders": orders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bvh")
    parser.add_argument("out")
    parser.add_argument(
        "--face-names", help="a file of face channel names, one per line"
    )
    parser.add_argument("--up", default="y", choices=["x", "y", "z"])
    args = parser.parse_args()
    with open(args.bvh) as source:
        skeleton = parse_hierarchy(source.read())
    skeleton["up"] = args.up
    skeleton["face"] = []
    if args.face_names:
        with open(args.face_names) as names:
            skeleton["face"] = [line.strip() for line in names if line.strip()]
    with open(args.out, "w") as out:
        json.dump(skeleton, out)
    print(
        f"{len(skeleton['joints'])} joints, {len(skeleton['face'])} face channels -> {args.out}"
    )


if __name__ == "__main__":
    main()
