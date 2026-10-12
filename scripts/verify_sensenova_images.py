# SPDX-License-Identifier: Apache-2.0
"""Verify SenseNova Images responses and input-token usage contracts."""

import argparse
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--image", type=Path, help="Optional reference image for I2I")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.n < 1 or args.steps < 1:
        parser.error("n and steps must be positive")
    try:
        width, height = map(int, args.size.split("x"))
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError
    except ValueError:
        parser.error("size must be WIDTHxHEIGHT with positive multiples of 32")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "SenseNova Images response and usage; not tokenizer ground truth",
        "not_covered": ["Chat usage", "performance benchmarks", "image edit quality"],
        "results": [],
    }

    def request_case(name, route, body, content_type, expected_count):
        started = time.monotonic()
        record = {"case": name, "passed": False}
        try:
            headers = {"Content-Type": content_type}
            if os.environ.get("OPENAI_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
            req = urllib.request.Request(
                args.base_url.rstrip("/") + route,
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                raw = response.read()
                record["http_status"] = response.status
            # Keep image payloads as PNGs, not redundant giant JSON/base64 files.
            data = json.loads(raw)
            assert isinstance(data.get("created"), int), "Missing integer created"
            images = data.get("data")
            assert (
                isinstance(images, list) and len(images) == expected_count
            ), "Unexpected image count"
            width, height = map(int, args.size.split("x"))
            usage = data.get("usage")
            assert isinstance(usage, dict), "Missing Images usage"
            details = usage.get("input_tokens_details")
            assert isinstance(details, dict), "Missing input_tokens_details"
            for key in ("text_tokens", "image_tokens"):
                value = details.get(key)
                assert type(value) is int and value >= 0, f"Invalid {key}"
            assert usage.get("input_tokens") == details["text_tokens"]
            assert usage.get("total_tokens") == details["text_tokens"]
            assert usage.get("image_count") == expected_count
            assert (usage.get("width"), usage.get("height")) == (width, height)
            if name == "t2i":
                assert details["image_tokens"] == 0
            else:
                assert details["image_tokens"] > 0
            for index, item in enumerate(images):
                assert item.get("output_format") == "png", "Missing PNG output_format"
                png = base64.b64decode(item["b64_json"], validate=True)
                with Image.open(io.BytesIO(png)) as image:
                    image.load()
                    assert image.format == "PNG", "Output is not PNG"
                    assert image.size == (width, height), "Unexpected output dimensions"
                (out / f"{name}-{index}.png").write_bytes(png)
            record["image_count"] = len(images)
            record["response_metadata"] = {
                key: value for key, value in data.items() if key != "data"
            }
            record["passed"] = True
        except urllib.error.HTTPError as error:
            record["http_status"] = error.code
            record["error"] = error.read().decode("utf-8", errors="replace")[:4000]
        except (AssertionError, OSError, ValueError, TypeError, KeyError) as error:
            record["error"] = str(error)
        record["client_e2e_seconds"] = time.monotonic() - started
        summary["results"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    payload = {
        "model": args.model,
        "prompt": "A red ceramic mug on a plain white table, studio photograph.",
        "size": args.size,
        "n": args.n,
        "seed": 42,
        "num_inference_steps": args.steps,
        "guidance_scale": 4.0,
        "response_format": "b64_json",
    }
    request_case(
        "t2i",
        "/v1/images/generations",
        json.dumps(payload).encode(),
        "application/json",
        args.n,
    )
    if args.image:
        boundary = "omni-" + uuid.uuid4().hex
        fields = dict(
            payload,
            n=args.n,
            img_cfg_scale=1.0,
            prompt="Keep the subject, composition and lighting unchanged. Make the background light blue.",
        )
        parts = []
        for key, value in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="reference.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
        )
        with Image.open(args.image) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        parts += [buffer.getvalue(), f"\r\n--{boundary}--\r\n".encode()]
        request_case(
            "i2i",
            "/v1/images/edits",
            b"".join(parts),
            f"multipart/form-data; boundary={boundary}",
            args.n,
        )
    summary["images_regression_passed"] = all(
        result["passed"] for result in summary["results"]
    )
    summary["passed"] = summary["images_regression_passed"]
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if summary["images_regression_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
