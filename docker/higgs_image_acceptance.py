"""GPU acceptance for the Higgs TTS image.

Four deploys in a row shipped an image that could not serve a request. Every
one of them was found by production, not by this script, and the reason was
always the same: the script exercised a path the deployment does not take.

The rule this file now follows is that acceptance drives
``higgs_tts_actor`` itself -- the exact function dramatiq invokes -- with the
payload shapes the gateway actually sends, and asserts on what the actor
hands to ``backend_callback``. Nothing here reimplements a step the actor
performs.

Concretely, the earlier version of this file could not have caught the
ffmpeg/av outage even in principle:

  * it hardcoded ``response_format: "wav"`` and wrote the result with the
    stdlib ``wave`` module, so it never reached ``_transcode_to_mp3`` --
    which shells out to the ``ffmpeg`` binary, absent from the image;
  * it took references as local ``.wav`` paths, which ``soundfile`` decodes
    natively, so it never reached the ``av`` decoder the HTTP(S)-URL
    references in production require.

Both gaps were format choices, not oversights: at each point the test picked
the one input shape that avoids the codec stack. So phase 2 below serves its
references over real HTTP and asserts on the produced ``.mp3``.

Phases:

1. Import the real service entrypoint (``background.workers.higgs``) in a
   subprocess. Import alone proves the module graph resolves -- it does not
   prove a request can be served, which is why it is not sufficient on its
   own.
2. Run three real jobs through ``higgs_tts_actor.fn``: plain, single-reference
   clone from an HTTP URL, and a three-source mix. Each must reach
   ``backend_callback`` with two ``filepath`` outputs and a decodable mp3.
   References are production-sized, not token-sized -- see ``_REF_SECONDS``.

Set ``BACKGROUND_DIR`` to point at the service source; phase 1 and 2 are
skipped when it is absent.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
import wave

BACKGROUND_DIR = os.environ.get("BACKGROUND_DIR", "/autodl-fs/data/prod/background")
MODEL = os.environ.get("HIGGS_MODEL_PATH", "/root/models/higgs-tts-3-4b")
TEXT = "你好，这是推理端镜像的验收测试，音色融合功能已经就绪。"

# The module reads these at import time; nothing in this script talks to OSS,
# and the fake callback in phase 2 replaces the only code that would.
PLACEHOLDER_ENV = {
    "OSS_ACCESS_KEY_ID": "placeholder",
    "OSS_ACCESS_KEY_SECRET": "placeholder",
    "OSS_GLOBAL_ACCESS_KEY_ID": "placeholder",
    "OSS_GLOBAL_ACCESS_KEY_SECRET": "placeholder",
}


def _service_env() -> dict:
    return {**os.environ, **PLACEHOLDER_ENV, "PYTHONPATH": BACKGROUND_DIR}


def check_service_entrypoint() -> None:
    """Phase 1: import what the deployment imports, in a throwaway process.

    A subprocess so the pipeline this boots releases the GPU before phase 2
    starts its own.
    """
    print("importing background.workers.higgs (boots the pipeline)...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", "import background.workers.higgs"],
        cwd=BACKGROUND_DIR,
        env=_service_env(),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        raise RuntimeError(
            "the deployment's own worker entrypoint does not import:\n" + tail
        )
    print(f"SERVICE_ENTRYPOINT_OK ({time.time() - t0:.1f}s)", flush=True)


def check_real_jobs() -> None:
    """Phase 2: re-exec this file inside the service's import context."""
    print("running real actor jobs...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--actor-phase"],
        cwd=BACKGROUND_DIR,
        env=_service_env(),
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError("real actor jobs failed (see output above)")
    print(f"REAL_JOBS_OK ({time.time() - t0:.1f}s)", flush=True)


# --------------------------------------------------------------------------
# phase 2 body -- runs in the subprocess started by check_real_jobs()
# --------------------------------------------------------------------------


def _probe_mp3(path: str) -> float:
    """Duration in seconds via ffprobe, which also proves the file decodes.

    Deliberately ffprobe and not a Python mp3 reader: the deployment's own
    transcode step is a subprocess call to the ffmpeg binary, so acceptance
    should fail the same way the deployment does when that binary is absent.
    """
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name:format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    codec, duration = out[0], float(out[1])
    if codec != "mp3":
        raise RuntimeError(f"{path}: expected mp3 stream, got {codec!r}")
    return duration


# References must be long enough to cost what production's cost. Peak encoder
# allocation scales linearly with reference length -- 89 MiB at 5 s, 533 MiB at
# 30 s -- so a short reference exercises the clone path while consuming almost
# none of the audio_encoder stage's GPU budget, and a GPU OOM that production
# hits on every clone request cannot reproduce here. Above
# HIGGS_REF_TRIM_SECONDS this also exercises the split-and-fuse path, which a
# short reference skips entirely.
_REF_SECONDS = int(os.environ.get("HIGGS_ACCEPTANCE_REF_SECONDS", "40"))


def _derive_reference_mp3(src_wav: str, dst_mp3: str, factor: float) -> None:
    """Build a distinct-timbre reference of _REF_SECONDS from real speech.

    Pitch-shifted while holding tempo, so fusion sees genuinely different
    voices, and looped up to length, so the encoder sees a production-sized
    input -- both without needing fixture audio staged on the machine.
    """
    with wave.open(src_wav) as wf:
        src_seconds = wf.getnframes() / wf.getframerate()
    loops = max(0, math.ceil(_REF_SECONDS / max(src_seconds, 0.1)) - 1)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", str(loops), "-i", src_wav,
            "-t", str(_REF_SECONDS),
            "-af", f"asetrate=24000*{factor},aresample=24000,atempo={1 / factor:.6f}",
            "-ar", "24000", "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "192k",
            dst_mp3,
        ],
        check=True,
        capture_output=True,
    )


def _actor_phase() -> int:
    import functools
    import hashlib
    import http.server
    import shutil
    import socketserver
    import tempfile
    import threading
    import uuid

    from background.tasks import higgs_tts_actor as actor_mod

    recorded: list[dict] = []

    def fake_callback(**kwargs):
        # The actor's failure branch also calls this -- with outputs=[] -- and
        # does not re-raise, so "no exception" is not a pass. Everything the
        # actor reports gets recorded and asserted on below.
        recorded.append(kwargs)

    actor_mod.backend_callback = fake_callback

    results_dir = tempfile.mkdtemp(prefix="higgs_acceptance_")
    serve_dir = tempfile.mkdtemp(prefix="higgs_refs_")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=serve_dir)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    httpd.allow_reuse_address = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    print(f"serving references at {base_url} from {serve_dir}", flush=True)

    # The actor hands cleanup of its work dir to a daemon thread that waits
    # for the OSS upload to consume the files. The fake callback above never
    # uploads, so that thread waits out its timeout and the process exits
    # first -- leaving a work dir per job under TMP_PATH. Harmless at runtime,
    # but this machine gets snapshotted into the image, so track and remove
    # them here rather than shipping the litter.
    work_dirs: list[str] = []

    def run_job(label: str, payload: dict) -> str:
        task_id = f"acc-{label}-{uuid.uuid4().hex[:8]}"
        work_dirs.append(os.path.join(actor_mod.TMP_PATH, task_id))
        recorded.clear()
        t0 = time.time()
        actor_mod.higgs_tts_actor.fn(
            task_id,
            {"api_payload": payload, "backend_url": None, "callback_path": None},
        )
        elapsed = time.time() - t0

        if not recorded:
            raise RuntimeError(f"{label}: actor never called backend_callback")
        outputs = recorded[-1].get("outputs") or []
        if len(outputs) != 2:
            raise RuntimeError(
                f"{label}: actor reported failure -- outputs={outputs!r} "
                f"callback={recorded[-1]!r}"
            )
        mp3 = outputs[0]["data"]
        duration = _probe_mp3(mp3)
        if duration < 0.5:
            raise RuntimeError(f"{label}: mp3 is only {duration:.2f}s")

        # Copy out before the actor's reaper thread removes the work dir.
        kept = os.path.join(results_dir, f"{label}.mp3")
        shutil.copy2(mp3, kept)
        wav = os.path.join(os.path.dirname(mp3), "output_audio.wav")
        if os.path.exists(wav):
            shutil.copy2(wav, os.path.join(results_dir, f"{label}.wav"))
        digest = hashlib.md5(open(kept, "rb").read()).hexdigest()
        print(
            f"  {label:16s} OK  {duration:5.2f}s audio  {os.path.getsize(kept):>8d}B mp3"
            f"  {elapsed:6.1f}s wall  md5={digest[:8]}",
            flush=True,
        )
        return digest

    try:
        # 1. Plain. Reaches _transcode_to_mp3, i.e. the ffmpeg binary.
        plain_digest = run_job("plain", {"text": TEXT, "voice": "default"})

        # Bootstrap references from that result rather than shipping fixture
        # audio, so this runs on a machine with nothing staged on disk.
        boot_wav = os.path.join(results_dir, "plain.wav")
        if not os.path.exists(boot_wav):
            raise RuntimeError("plain job produced no wav to derive references from")
        refs = {}
        for name, factor in (("low", 0.85), ("mid", 1.0), ("high", 1.18)):
            path = os.path.join(serve_dir, f"ref_{name}.mp3")
            _derive_reference_mp3(boot_wav, path, factor)
            refs[name] = f"{base_url}/ref_{name}.mp3"
        print(f"  references: {_REF_SECONDS}s each, 3 timbres", flush=True)

        # 2. Single-reference clone from an HTTP URL. Reaches the av decoder.
        clone_digest = run_job(
            "clone_url", {"text": TEXT, "reference_audio": refs["low"]}
        )

        # 3. Three-source mix -- the fork's reason for existing, and the
        # heaviest thing the audio_encoder stage is asked to do: three
        # references of _REF_SECONDS each, encoded max_concurrency-wide.
        mix_digest = run_job(
            "mix_url",
            {
                "text": TEXT,
                "source_urls": [refs["low"], refs["mid"], refs["high"]],
                "weights": [0.5, 0.3, 0.2],
            },
        )

        # A reference that fails to load can be dropped without failing the
        # request, in which case both jobs above degrade to plain synthesis
        # and every assertion so far still passes -- the mp3 is real, it just
        # is not a clone. Generation is deterministic for a fixed prompt, so
        # matching the plain digest means the reference had no effect.
        # (Equal byte counts do not: these are CBR mp3s, so identical
        # durations give identical sizes.)
        for label, digest in (("clone_url", clone_digest), ("mix_url", mix_digest)):
            if digest == plain_digest:
                raise RuntimeError(
                    f"{label}: output is identical to plain synthesis, so the "
                    f"reference audio was accepted and then ignored"
                )
    except BaseException:
        # Leave the artifacts behind only when there is something to look at.
        print(f"FAILED -- artifacts kept in {results_dir}", flush=True)
        raise
    finally:
        httpd.shutdown()
        for path in (*work_dirs, serve_dir):
            shutil.rmtree(path, ignore_errors=True)

    shutil.rmtree(results_dir, ignore_errors=True)
    return 0


def main() -> int:
    if not os.path.isdir(BACKGROUND_DIR):
        print(f"SKIPPED (no service source at {BACKGROUND_DIR})", flush=True)
        return 0
    check_service_entrypoint()
    check_real_jobs()
    print("ACCEPTANCE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    if "--actor-phase" in sys.argv:
        sys.exit(_actor_phase())
    sys.exit(main())
