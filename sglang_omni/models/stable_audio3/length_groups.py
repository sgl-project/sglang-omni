# SPDX-License-Identifier: Apache-2.0
"""Split a Stable Audio 3 batch into runs of similar length, so a short request is not padded to a long one."""

import math


def plan_length_groups(
    latent_lengths: list[int], overhead_latents: int
) -> list[list[int]]:
    """Cut a batch, sorted by length, into runs that each pad only to their own longest request.

    The cut minimizes the padded latents plus a fixed overhead per run, so a batch of equal lengths stays whole. Returns indices into latent_lengths.
    """
    order = sorted(range(len(latent_lengths)), key=lambda index: latent_lengths[index])
    best_cost = [0] + [math.inf] * len(order)
    run_start = [0] * (len(order) + 1)
    for end in range(1, len(order) + 1):
        for start in range(end):
            cost = (
                best_cost[start]
                + overhead_latents
                + (end - start) * latent_lengths[order[end - 1]]
            )
            if cost < best_cost[end]:
                best_cost[end] = cost
                run_start[end] = start
            else:
                pass
    groups = []
    end = len(order)
    while end > 0:
        groups.append(order[run_start[end] : end])
        end = run_start[end]
    return groups[::-1]
