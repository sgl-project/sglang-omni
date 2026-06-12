# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir aiohttp==3.13.5 openai==2.6.1
RUN mkdir -p /workspace/benchmarks/eval /workspace/benchmarks/tts_serving /var/benchmark/out

COPY benchmarks/eval/benchmark_tts_serving.py /workspace/benchmarks/eval/benchmark_tts_serving.py
COPY benchmarks/tts_serving /workspace/benchmarks/tts_serving

RUN useradd --create-home --uid 10001 benchmark \
    && chown -R benchmark:benchmark /workspace /var/benchmark

USER benchmark

ENTRYPOINT ["python", "-m", "benchmarks.eval.benchmark_tts_serving"]
