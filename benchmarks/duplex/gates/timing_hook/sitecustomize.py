"""Measurement-only stage timing hook for the duplex gates (not part of the server).

Put this directory first on PYTHONPATH and set STAGE_TIMING_DIR; without the variable the
module does nothing. Every process of the server (the main process and one per stage)
appends JSON lines {"t": start, "dt": seconds, "kind": ..., "n": batch size} to
<dir>/<pid>.jsonl. Records are flushed every second because the gates stop the server with
SIGKILL. "t" is time.perf_counter(), the same clock as the recorder traces' "time_s".

Recorded kinds:
  hooks:<Hooks>          SessionScheduler.compute_batch (perception, speech), n = sessions in the batch
  open:<Hooks>           SessionScheduler.open_session
  perc:append[_batch]    PerceptionHooks.append / append_batch
  perc:image, perc:audio one frame's image encoding / one batched streaming audio encoder call
  omni:next_batch:<mode> also carries "tokens" (new prompt tokens of an extend batch) and "max_seq" / "sum_seq" (context lengths)
  omni:run_batch:<mode>  OmniScheduler (thinker, talker) batch, plus omni:next_batch / launch / resolve
  engine:<method>        SGLang Scheduler.run_batch / process_batch_result
  gc                     every garbage-collector pause over 5 ms (n = generation)
  mem                    peak reserved / allocated CUDA memory of the process, sampled every second

Session-stage and perception calls are synchronized after the call so that "dt" includes the
GPU work; STAGE_TIMING_LIGHT=1 turns that synchronization off.
"""

import gc
import importlib.abc
import json
import os
import sys
import threading
import time

OUT_DIR = os.environ.get("STAGE_TIMING_DIR")
LIGHT = os.environ.get("STAGE_TIMING_LIGHT") == "1"
GC_PAUSE_MIN_S = 0.005


class Recorder:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{os.getpid()}.jsonl")
        self.records = []
        self.lock = threading.Lock()
        self.gc_start = {}
        threading.Thread(target=self.flush_forever, daemon=True).start()

    def emit(self, record):
        with self.lock:
            self.records.append(record)

    def timed(self, kind, t0, n, **extra):
        record = {"t": t0, "dt": time.perf_counter() - t0, "kind": kind, "n": n}
        record.update(extra)
        self.emit(record)

    def sample_memory(self):
        torch = cuda_torch()
        if torch is None:
            return
        else:
            self.emit(
                {
                    "t": time.perf_counter(),
                    "dt": 0.0,
                    "kind": "mem",
                    "n": 0,
                    "max_reserved_mb": int(torch.cuda.max_memory_reserved() >> 20),
                    "max_alloc_mb": int(torch.cuda.max_memory_allocated() >> 20),
                }
            )

    def flush_forever(self):
        while True:
            time.sleep(1.0)
            try:
                self.sample_memory()
            except Exception as exc:  # the sampler must never stop the flushing
                sys.stderr.write(f"[stage-timing] memory sample failed: {exc!r}\n")
            with self.lock:
                batch, self.records = self.records, []
            if batch:
                with open(self.path, "a") as f:
                    for record in batch:
                        f.write(json.dumps(record) + "\n")
            else:
                pass

    def on_gc(self, phase, info):
        ident = threading.get_ident()
        if phase == "start":
            self.gc_start[ident] = time.perf_counter()
            return
        else:
            t0 = self.gc_start.pop(ident, None)
        if t0 is not None and time.perf_counter() - t0 > GC_PAUSE_MIN_S:
            self.timed(
                "gc",
                t0,
                info.get("generation"),
                collected=info.get("collected"),
                thread=threading.current_thread().name[:24],
            )
        else:
            pass


def cuda_torch():
    """torch once it is fully imported and CUDA is initialized in this process, else None (never initializes CUDA)."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    return torch if is_initialized is not None and is_initialized() else None


def synchronize():
    torch = cuda_torch()
    if not LIGHT and torch is not None:
        torch.cuda.synchronize()
    else:
        pass


def batch_mode(batch):
    mode = getattr(batch, "forward_mode", "")
    return getattr(mode, "name", str(mode).split(".")[-1])


def batch_size(batch):
    size = getattr(batch, "batch_size", None)
    return size() if callable(size) else -1


def wrap_method(recorder, cls, name, kind_fn, size_fn, sync=False):
    original = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            if sync:
                synchronize()
            else:
                pass
            recorder.timed(
                kind_fn(self, *args, **kwargs), t0, size_fn(self, *args, **kwargs)
            )

    setattr(cls, name, wrapper)


def patch_session(recorder, module):
    cls = module.SessionScheduler

    def hooks_name(self, *args, **kwargs):
        return type(self.session_hooks).__name__

    wrap_method(
        recorder,
        cls,
        "open_session",
        lambda self, *a, **k: "open:" + hooks_name(self),
        lambda self, *a, **k: 1,
        sync=True,
    )
    wrap_method(
        recorder,
        cls,
        "compute_batch",
        lambda self, *a, **k: "hooks:" + hooks_name(self),
        lambda self, payloads, *a, **k: len(payloads),
        sync=True,
    )


def patch_engine(recorder, module):
    cls = module.Scheduler
    wrap_method(
        recorder,
        cls,
        "run_batch",
        lambda self, batch, *a, **k: "engine:run_batch:" + batch_mode(batch),
        lambda self, batch, *a, **k: batch_size(batch),
    )
    wrap_method(
        recorder,
        cls,
        "process_batch_result",
        lambda self, batch, *a, **k: "engine:process_batch_result",
        lambda self, batch, *a, **k: batch_size(batch),
    )


def token_counts(batch):
    """New prompt tokens of an extend batch and the context lengths of its requests (host-side fields only)."""
    counts = {}
    extend_tokens = getattr(batch, "extend_num_tokens", None)
    if isinstance(extend_tokens, int):
        counts["tokens"] = extend_tokens
    else:
        pass
    seq_lens = getattr(batch, "seq_lens_cpu", None)
    if seq_lens is not None and len(seq_lens):
        counts["max_seq"] = int(seq_lens.max())
        counts["sum_seq"] = int(seq_lens.sum())
    else:
        pass
    return counts


def patch_omni(recorder, module):
    cls = module.OmniScheduler
    for name, label in (
        ("run_batch", "run_batch"),
        ("run_batch_launch", "launch"),
        ("run_batch_resolve", "resolve"),
    ):
        if name in cls.__dict__:
            wrap_method(
                recorder,
                cls,
                name,
                lambda self, batch, *a, label=label, **k: f"omni:{label}:"
                + batch_mode(batch),
                lambda self, batch, *a, **k: batch_size(batch),
            )
        else:
            pass
    original_next = cls.get_next_batch_to_run

    def get_next_batch_to_run(self, *args, **kwargs):
        t0 = time.perf_counter()
        batch = original_next(self, *args, **kwargs)
        if batch is not None:
            recorder.timed(
                "omni:next_batch:" + batch_mode(batch),
                t0,
                batch_size(batch),
                q=len(self.waiting_queue),
                **token_counts(batch),
            )
        else:
            pass
        return batch

    cls.get_next_batch_to_run = get_next_batch_to_run


def patch_native_stages(recorder, module):
    cls = module.PerceptionHooks
    if "append_batch" in cls.__dict__:
        wrap_method(
            recorder,
            cls,
            "append_batch",
            lambda self, *a, **k: "perc:append_batch",
            lambda self, *a, **k: len(a[0]) if a else len(k.get("appends", ())),
            sync=True,
        )
    else:
        pass
    if "append" in cls.__dict__:
        wrap_method(
            recorder,
            cls,
            "append",
            lambda self, *a, **k: "perc:append",
            lambda self, *a, **k: 1,
            sync=True,
        )
    else:
        pass


def patch_streaming_perception(recorder, module):
    for cls in vars(module).values():
        if isinstance(cls, type) and "encode_image" in cls.__dict__:
            wrap_method(
                recorder,
                cls,
                "encode_image",
                lambda self, *a, **k: "perc:image",
                lambda self, *a, **k: 1,
                sync=True,
            )
        else:
            pass


def patch_audio_encoder(recorder, module):
    wrap_method(
        recorder,
        module.MiniCPMOAudioEncoder,
        "forward_streaming_batch",
        lambda self, *a, **k: "perc:audio",
        lambda self, chunks, *a, **k: len(chunks),
        sync=True,
    )


TARGETS = {
    "sglang_omni.models.minicpm_o.components.streaming_perception": patch_streaming_perception,
    "sglang_omni.models.minicpm_o.components.audio_encoder": patch_audio_encoder,
    "sglang_omni.scheduling.session": patch_session,
    "sglang.srt.managers.scheduler": patch_engine,
    "sglang_omni.scheduling.omni_scheduler": patch_omni,
    "sglang_omni.models.minicpm_o.native_stages": patch_native_stages,
}


class PatchingLoader(importlib.abc.Loader):
    def __init__(self, recorder, inner, name):
        self.recorder, self.inner, self.name = recorder, inner, name

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            TARGETS[self.name](self.recorder, module)
            sys.stderr.write(
                f"[stage-timing] patched {self.name} in pid {os.getpid()}\n"
            )
        except (
            Exception
        ) as exc:  # a missing symbol on another tree must not stop the server
            sys.stderr.write(f"[stage-timing] patch failed for {self.name}: {exc!r}\n")


class PatchingFinder(importlib.abc.MetaPathFinder):
    def __init__(self, recorder):
        self.recorder = recorder

    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        else:
            pass
        for finder in sys.meta_path:
            if finder is self:
                continue
            else:
                spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = PatchingLoader(self.recorder, spec.loader, name)
                return spec
            else:
                pass
        return None


if OUT_DIR:
    RECORDER = Recorder(OUT_DIR)
    gc.callbacks.append(RECORDER.on_gc)
    sys.meta_path.insert(0, PatchingFinder(RECORDER))
else:
    pass
