# SPDX-License-Identifier: Apache-2.0
"""Test-process observations; loaded by the smoke fixture's sitecustomize."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path

SCHEDULER = "sglang_omni.scheduling.omni_scheduler"
CACHE = "sglang.srt.mem_cache.common"


def _write(event, *, scheduler=None, **data):
    module = sys.modules.get(SCHEDULER)
    stage = module._get_active_stage() if module is not None else None
    if scheduler is not None:
        stage = stage or getattr(scheduler, "_dflash_test_stage", None)
        if stage:
            scheduler._dflash_test_stage = stage
    record = dict(
        event=event,
        stage=stage,
        pid=os.getpid(),
        tp_rank=getattr(scheduler, "tp_rank", None),
        tp_size=getattr(scheduler, "tp_size", None),
        monotonic_ns=time.monotonic_ns(),
        **data,
    )
    root = Path(os.environ["OMNI_DFLASH_TEST_TRACE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"events.{os.getpid()}.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")


def _wait_for_resume(root):
    deadline = time.monotonic() + 30
    while not (root / "resume_request").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("DFlash smoke cancellation barrier timed out")
        time.sleep(0.01)


def _patch_cache(module):
    original = module.release_kv_cache

    @functools.wraps(original)
    def release(req, *args, **kwargs):
        before = req.req_pool_idx
        result = original(req, *args, **kwargs)
        _write(
            "release",
            request_id=req.rid,
            req_pool_idx_before=before,
            req_pool_idx_after=req.req_pool_idx,
        )
        return result

    module.release_kv_cache = release


def _patch_scheduler(module):
    cls = module.OmniScheduler
    original_stream = cls.stream_output
    original_put = cls._put_stream_messages
    original_forward = cls._run_speculative_batch
    original_prefill = cls.get_new_batch_prefill
    original_abort = cls.abort

    @functools.wraps(original_stream)
    def stream(self, reqs, return_logprob=False, skip_req=None):
        seen = self.__dict__.setdefault("_dflash_test_terminals", set())
        terminals = []
        for req in reqs:
            if req is skip_req or not req.finished() or req.rid in seen:
                continue
            seen.add(req.rid)
            terminals.append(
                dict(
                    request_id=req.rid,
                    ids=list(req.output_ids_through_stop),
                    eos_token_ids=sorted(req.eos_token_ids or []),
                    tokenizer_eos_token_id=getattr(req.tokenizer, "eos_token_id", None),
                    finished_reason=req.finished_reason.to_json(),
                    aborted=req.rid in self._aborted_request_ids
                    or isinstance(req.finished_reason, module.FINISH_ABORT),
                    spec_verify_ct=req.spec_verify_ct,
                    spec_num_correct_drafts=req.spec_num_correct_drafts,
                    req_pool_idx=req.req_pool_idx,
                )
            )
        result = original_stream(self, reqs, return_logprob, skip_req)
        # note(wenyao): Terminal evidence must follow the final stream rows on
        # every TP rank; capture IDs before cleanup but publish after emission.
        for record in terminals:
            _write("terminal", scheduler=self, **record)
        return result

    @functools.wraps(original_put)
    def put(self, request_id, messages):
        messages = list(messages)
        result = original_put(self, request_id, messages)
        for message in messages:
            token = (message.metadata or {}).get("token_id")
            if token is not None:
                _write(
                    "stream",
                    scheduler=self,
                    request_id=request_id,
                    target=message.target,
                    token_id=token,
                )
        return result

    @functools.wraps(original_forward)
    def forward(self, batch):
        is_decode = batch.forward_mode.is_decode()
        if is_decode:
            _write(
                "verify_start",
                scheduler=self,
                request_ids=[req.rid for req in batch.reqs],
            )
        result = original_forward(self, batch)
        root = Path(os.environ["OMNI_DFLASH_TEST_TRACE_DIR"])
        pause = root / "pause_request"
        if is_decode and pause.exists():
            request_id = pause.read_text().strip()
            if any(req.rid == request_id for req in batch.reqs) and not getattr(
                self, "_dflash_test_paused", False
            ):
                # note(wenyao): Pause before the scheduler commits the verified
                # result, making disconnect timing deterministic without changing kernels.
                self._dflash_test_paused = True
                _write("verify_paused", scheduler=self, request_id=request_id)
                _wait_for_resume(root)
        return result

    @functools.wraps(original_prefill)
    def prefill(self, running_batch):
        root = Path(os.environ["OMNI_DFLASH_TEST_TRACE_DIR"])
        pause = root / "pause_chunked_request"
        req = self.chunked_req
        if (
            req is not None
            and req.req_pool_idx is not None
            and not getattr(self, "_dflash_test_chunked_paused", False)
            and pause.exists()
            and req.rid == pause.read_text().strip()
        ):
            batches = {
                "running_argument": running_batch,
                "running": self.running_batch,
                "cur": self.cur_batch,
                "last": self.last_batch,
                "async_pending": self._async_pending_batch(),
            }
            aliases = [
                name
                for name, batch in batches.items()
                if batch is not None and any(item is req for item in batch.reqs)
            ]
            if not aliases:
                # note(wenyao): Upstream has filtered the completed chunk from
                # its batches; hold the parked request before building its next chunk.
                self._dflash_test_chunked_paused = True
                _write(
                    "chunked_paused",
                    scheduler=self,
                    request_id=req.rid,
                    req_pool_idx=req.req_pool_idx,
                    prefix_tokens=len(req.prefix_indices),
                    batch_aliases=aliases,
                )
                _wait_for_resume(root)
        return original_prefill(self, running_batch)

    @functools.wraps(original_abort)
    def abort(self, request_id, **kwargs):
        batches = (self.running_batch, self.cur_batch, self.last_batch)
        found = (
            any(
                req.rid == request_id
                for batch in batches
                if batch is not None
                for req in batch.reqs
            )
            or getattr(self.chunked_req, "rid", None) == request_id
        )
        result = original_abort(self, request_id, **kwargs)
        _write("abort", scheduler=self, request_id=request_id, found=found)
        return result

    cls.stream_output = stream
    cls._put_stream_messages = put
    cls._run_speculative_batch = forward
    cls.get_new_batch_prefill = prefill
    cls.abort = abort


class _Loader:
    def __init__(self, original, patch):
        self.original = original
        self.patch = patch

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        self.patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        patch = {SCHEDULER: _patch_scheduler, CACHE: _patch_cache}.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, patch)
        return spec


def install():
    if os.environ.get("OMNI_DFLASH_TEST_TRACE_DIR") and not any(
        isinstance(finder, _Finder) for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _Finder())
