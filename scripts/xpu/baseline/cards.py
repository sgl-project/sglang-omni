# SPDX-License-Identifier: Apache-2.0
"""Which physical cards a run may take, and whether another process holds them before or after it."""

from __future__ import annotations

import os
import time

import torch


class CardsBusy(RuntimeError):
    """Raised instead of measuring when another process holds a card, so it is not read as a regression."""


def visible_cards() -> list[int]:
    """Physical ids of the cards this process may use, in torch's order: an inherited mask applies ascending."""
    mask = os.environ.get("ZE_AFFINITY_MASK", "").strip()
    if mask:
        return sorted({int(part) for part in mask.split(",") if part.strip()})
    else:
        return list(range(torch.xpu.device_count()))


def cards_in_use(cards: str) -> dict[str, float]:
    """MiB already allocated on each physical card, device-wide so a neighbour's work shows."""
    visible = visible_cards()
    index = {card: visible.index(int(card)) for card in cards.split(",")}
    used: dict[str, float] = {}
    for card, device in index.items():
        free, total = torch.xpu.mem_get_info(device)
        used[card] = round((total - free) / 1024**2, 1)
    return used


def allocate_cards(pool: list[int], count: int, taken: set[int]) -> str:
    """The lowest unused cards of the pool as a ZE_AFFINITY_MASK, so a model lands on the same cards every run."""
    free = [card for card in pool if card not in taken]
    if len(free) < count:
        raise AssertionError(
            f"need {count} card(s), pool {pool} has {len(free)} free ({sorted(taken)} in use)"
        )
    else:
        chosen = free[:count]
    taken.update(chosen)
    return ",".join(str(card) for card in chosen)


def idle_before_launch(layout: str, idle_mib: float, wait_s: int) -> dict[str, float]:
    """Card memory once every card is idle, waiting up to wait_s; raises CardsBusy past that."""
    deadline = time.time() + wait_s
    used = cards_in_use(layout)
    while True:
        noisy = {card: mib for card, mib in used.items() if mib > idle_mib}
        if not noisy:
            return used
        elif time.time() >= deadline:
            raise CardsBusy(
                f"card(s) not idle before launch: {noisy} MiB in use, over the "
                f"{idle_mib:g} MiB bar; another process on the card makes this a "
                "different machine"
            )
        else:
            print(f"  waiting for {noisy} MiB to clear", flush=True)
        time.sleep(30)
        used = cards_in_use(layout)


def idle_after_run(layout: str, idle_mib: float, wait_s: int = 60) -> dict[str, float]:
    """Card memory once this run's own servers have released it; raises CardsBusy when another process stays."""
    deadline = time.time() + wait_s
    used = cards_in_use(layout)
    while any(mib > idle_mib for mib in used.values()) and time.time() < deadline:
        time.sleep(5)
        used = cards_in_use(layout)
    noisy = {card: mib for card, mib in used.items() if mib > idle_mib}
    if noisy:
        raise CardsBusy(
            f"card(s) still in use after this run's servers exited: {noisy} MiB, "
            f"over the {idle_mib:g} MiB bar; another process arrived during the run, so "
            "its numbers are not kept"
        )
    else:
        return used
