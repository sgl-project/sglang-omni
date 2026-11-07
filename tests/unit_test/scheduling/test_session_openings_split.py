# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge


def unit(session_id: str, *, aborted: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        rid=session_id,
        session=SimpleNamespace(session_id=session_id, streaming=True),
        to_finish=object() if aborted else None,
    )


def bridge(holding: set[str]) -> ARSessionBridge:
    records = {name: SimpleNamespace(holds_kv=True) for name in holding}
    scheduler = SimpleNamespace(
        tree_cache=SimpleNamespace(session_records=lambda: records)
    )
    return ARSessionBridge(scheduler, adapter=None)


def test_units_split_by_whether_their_session_holds_kv():
    queue = [unit("a"), unit("new"), unit("b"), unit("b2", aborted=True)]
    continuing, opening = bridge({"a", "b", "b2"}).split_openings(queue)
    assert [u.rid for u in continuing] == ["a", "b"]
    assert [u.rid for u in opening] == ["new", "b2"]


class Loop:
    get_next_batch_to_run = OmniScheduler.get_next_batch_to_run
    separate_session_openings = True
    last_prefill_continued_sessions = False
    chunked_req = None
    last_batch = None
    is_on_tick_grid = False

    def __init__(self, queue, holding):
        self.waiting_queue = queue
        self.session_bridge = bridge(holding)
        self.running_batch = SimpleNamespace(is_empty=lambda: True, batch_is_full=False)


def test_openings_and_continuing_units_alternate(monkeypatch):
    def upstream(self, running_batch, last_batch):
        taken = list(self.waiting_queue)
        self.waiting_queue = []
        return SimpleNamespace(
            batch_to_run=[u.rid for u in taken] or None, running_batch=running_batch
        )

    monkeypatch.setattr(
        omni_scheduler._Upstream,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        "get_next_batch_to_run",
        upstream,
        raising=False,
    )
    loop = Loop([unit("new"), unit("a"), unit("b")], {"a", "b"})
    assert loop.get_next_batch_to_run() == ["a", "b"]
    assert [u.rid for u in loop.waiting_queue] == ["new"]
    loop.waiting_queue += [unit("a"), unit("b")]
    assert loop.get_next_batch_to_run() == ["new"]
    assert loop.get_next_batch_to_run() == ["a", "b"]
    loop.separate_session_openings = False
    loop.waiting_queue = [unit("new"), unit("a")]
    assert loop.get_next_batch_to_run() == ["new", "a"]
