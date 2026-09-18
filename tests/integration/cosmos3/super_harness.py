# SPDX-License-Identifier: Apache-2.0
"""Shared SDK/HTTP transport and measurement helpers for opt-in Super runs."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import httpx


def transports():
    values = os.environ.get("COSMOS3_SUPER_TRANSPORTS", "sdk,http").split(",")
    if not values or any(v not in ("sdk", "http") for v in values):
        raise ValueError("COSMOS3_SUPER_TRANSPORTS must be sdk, http, or sdk,http")
    return list(dict.fromkeys(values))


def write_config(config, directory):
    """Preserve the public YAML schema while applying the test's placement."""
    import yaml

    kind = config.stages[0].name
    root = Path(__file__).resolve().parents[3]
    data = yaml.safe_load(
        (root / f"examples/configs/cosmos3_super_{kind}.yaml").read_text()
    )
    stage = config.stages[0]
    data["model_path"] = config.model_path
    data["stages"][kind].update(
        gpu=stage.gpu,
        runtime_gpu_ids=stage.runtime_gpu_ids,
        factory=stage.factory.model_dump(mode="json", exclude_none=True),
    )
    path = directory / "deployment.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@contextmanager
def measurements(directory, devices):
    """Sample device-wide memory, including native child workers and startup.

    Use an otherwise idle allocation. NVML polling is a sampled high-water mark,
    not an exact allocator peak. Errors are persisted, never reported as zero.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    records = []
    peaks = {}
    succeeded = False
    started = time.monotonic()

    def sample():
        while not stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={','.join(map(str, devices))}",
                        "--query-gpu=index,name,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                gpus = []
                for row in result.stdout.strip().splitlines():
                    index, name, used, total = [
                        value.strip() for value in row.split(",")
                    ]
                    used = float(used)
                    gpus.append(
                        {
                            "index": index,
                            "name": name,
                            "used_mib": used,
                            "total_mib": float(total),
                        }
                    )
                    peaks[index] = max(peaks.get(index, 0), used)
                if len(gpus) != len(devices):
                    raise ValueError(
                        "Memory sampler did not return every requested GPU"
                    )
                records.append({"elapsed_s": time.monotonic() - started, "gpus": gpus})
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                records.append({"error": str(exc)})
                return
            stop.wait(0.5)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
        succeeded = True
    finally:
        stop.set()
        thread.join(timeout=6)
        (directory / "memory.json").write_text(
            json.dumps(
                {
                    "scope": "device-wide sampled MiB; includes startup and inference",
                    "elapsed_s": time.monotonic() - started,
                    "sampled_peak_mib": peaks,
                    "samples": records,
                },
                indent=2,
            )
        )
        if succeeded:
            assert records and all("error" not in row for row in records), (
                "Memory sampling failed; inspect memory.json"
            )


@asynccontextmanager
async def deployment(config, transport, directory):
    from benchmarks.benchmarker.utils import managed_omni_server
    from tests.integration.cosmos3.test_super_gpu import _running

    directory.mkdir(parents=True, exist_ok=True)
    config_path = write_config(config, directory)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "transport": transport,
                "checkpoint": config.model_path,
                "gpu_ids": config.stages[0].runtime_gpu_ids,
                "native_revision": os.environ.get(
                    "COSMOS3_SUPER_NATIVE_REVISION", "unset"
                ),
                "omni_revision": (
                    await asyncio.to_thread(
                        subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
                    )
                ).strip(),
            },
            indent=2,
        )
    )
    with measurements(directory, config.stages[0].runtime_gpu_ids):
        if transport == "sdk":
            async with _running(config) as runner:
                yield SuperClient(transport, directory, runner=runner)
        else:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            with managed_omni_server(
                model_path=config.model_path,
                port=port,
                host="127.0.0.1",
                log_file=directory / "server.log",
                server_config=str(config_path),
                timeout=int(os.environ.get("COSMOS3_SUPER_STARTUP_TIMEOUT", "1800")),
                wait_for_gpu_release=False,
            ):
                async with httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{port}",
                    timeout=float(
                        os.environ.get("COSMOS3_SUPER_REQUEST_TIMEOUT", "1800")
                    ),
                ) as http:
                    yield SuperClient(transport, directory, http=http)


class SuperClient:
    def __init__(self, transport, directory, *, runner=None, http=None):
        self.transport, self.directory = transport, directory
        self.runner, self.http = runner, http
        self.sequence = 0

    async def completion(self, payload):
        import aiohttp

        try:
            async with asyncio.timeout(
                float(os.environ.get("COSMOS3_SUPER_REQUEST_TIMEOUT", "1800"))
            ):
                body = await self._completion(payload)
            if not body["choices"][0]["message"]["content"].strip():
                raise ValueError("Empty Reasoner response")
            return body
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            # Shared benchmark senders persist transport failures as failed samples.
            raise aiohttp.ClientError(str(exc)) from exc

    async def _completion(self, payload):
        if self.transport == "http":
            response = await self.http.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        from sglang_omni.client import Client, GenerateRequest, SamplingParams

        request = GenerateRequest(
            prompt={
                k: payload[k] for k in ("messages", "images", "videos") if k in payload
            },
            sampling=SamplingParams(
                temperature=payload["temperature"], max_new_tokens=payload["max_tokens"]
            ),
            stream=False,
        )
        chunks = [
            chunk async for chunk in Client(self.runner.coordinator).generate(request)
        ]
        assert len(chunks) == 1 and chunks[0].finish_reason in ("stop", "length")
        chunk = chunks[0]
        return {
            "choices": [{"message": {"content": chunk.text}}],
            "usage": chunk.usage.to_dict() if chunk.usage else {},
        }

    async def generate(self, inputs):
        started = time.monotonic()
        try:
            async with asyncio.timeout(
                float(os.environ.get("COSMOS3_SUPER_REQUEST_TIMEOUT", "1800"))
            ):
                return await self._generate(inputs)
        finally:
            with (self.directory / "requests.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {"inputs": inputs, "latency_s": time.monotonic() - started}
                    )
                    + "\n"
                )

    async def _generate(self, inputs):
        from sglang_omni.client import GenerateChunk

        if self.transport == "sdk":
            from tests.test_model.test_cosmos3_super_generator_ci import _generate

            return await _generate(self.runner, inputs)
        self.sequence += 1
        action = inputs.get("action_mode") in ("policy", "inverse_dynamics")
        image = inputs.get("num_frames") == 1 and not action
        suffix = ".json" if action else ".png" if image else ".mp4"
        path = self.directory / f"output-{self.sequence}{suffix}"
        if image:
            response = await self.http.post(
                "/v1/images/generations",
                json={
                    **inputs,
                    "response_format": "b64_json",
                    "output_format": "png",
                },
            )
            response.raise_for_status()
            path.write_bytes(
                base64.b64decode(response.json()["data"][0]["b64_json"], validate=True)
            )
        else:
            params = dict(inputs)
            prompt = params.pop("prompt", "")
            files = {}
            for key, field in (
                ("image_path", "input_reference"),
                ("video_path", "video_reference"),
            ):
                source = params.pop(key, None)
                if source:
                    files[field] = (
                        Path(source).name,
                        await asyncio.to_thread(Path(source).read_bytes),
                        mimetypes.guess_type(source)[0] or "application/octet-stream",
                    )
            fields = {"prompt": prompt, "extra_params": json.dumps(params)}
            if not action and "width" in params and "height" in params:
                fields["size"] = f"{params['width']}x{params['height']}"
            # Force multipart even for text-only video requests.
            parts = [(k, (None, v)) for k, v in fields.items()] + list(files.items())
            endpoint = "/v1/actions/generations" if action else "/v1/videos"
            response = await self.http.post(endpoint, files=parts)
            response.raise_for_status()
            body = response.json()
            if action:
                path.write_text(json.dumps(body))
            else:
                video_id = body["id"]
                while body["status"] not in ("completed", "failed", "cancelled"):
                    await asyncio.sleep(1)
                    response = await self.http.get(f"/v1/videos/{video_id}")
                    response.raise_for_status()
                    body = response.json()
                assert body["status"] == "completed", body
                response = await self.http.get(f"/v1/videos/{video_id}/content")
                response.raise_for_status()
                path.write_bytes(response.content)
        return GenerateChunk(
            request_id=str(self.sequence),
            finish_reason="stop",
            media=[{"path": str(path)}],
        )
