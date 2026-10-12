# Response/usage and Metrics acceptance scripts

Scripts use the current Python interpreter and write compact evidence into the
requested output directory. They do not install dependencies, launch models,
mock platform imports in native regression, or claim GPU/NPU validation from
CPU tests. API authentication uses OPENAI_API_KEY when set; metrics scraping
does not require an API key. Do not commit generated results.

## Model-free regression (Linux)

Run from the checkout being accepted, using its intended environment:

```bash
python scripts/verify_response_usage.py --repo "$PWD" --mode smoke --out-dir results/response-smoke
python scripts/verify_response_usage.py --repo "$PWD" --mode full --out-dir results/response-full
python scripts/verify_metrics.py --repo "$PWD" --mode smoke --out-dir results/metrics-smoke
python scripts/verify_metrics.py --repo "$PWD" --mode full --out-dir results/metrics-full
```

The Response full regression includes existing audio compatibility tests; a
known environment/baseline M4A failure must be recorded, not silently skipped.
Metrics full regression selects the existing engine snapshot test for the
formal branch or its older-branch equivalent, and records the exact pytest
command. Missing required tests are an error. Artifacts: pytest.log, junit.xml,
summary.json. Use python -m pip/python -m pytest in the same environment if
resolving dependencies; pip show from a different interpreter is not evidence.

## Live AR Chat response/usage

Start a supported AR model using its normal pipeline command. Use its served
model name, not an Images-only model:

```bash
python scripts/verify_model_response_usage.py --base-url http://127.0.0.1:8000 --model "$AR_MODEL" --out-dir results/live-response
```

This checks non-streaming usage, default streaming without usage and opt-in
terminal streaming usage, token arithmetic, finish/DONE ordering and optional
detail types. Add --expect-cached-details only when the backend actually
reports prompt_tokens_details.cached_tokens. Missing producer data is not
invented. Response arithmetic is not independent tokenizer ground truth;
reasoning/audio details require suitable models. No-card regression covers
field plumbing, but live usage data still requires an AR model.

## SenseNova Images compatibility

Use test/omni-response-usage-sensenova for the integrated SenseNova pipeline;
the formal main-based metrics branch does not acquire SenseNova support just
because this script is present. Start the service with its usual command and
add --enable-metrics when validating metrics.

```bash
python scripts/verify_sensenova_images.py --base-url http://127.0.0.1:8000 --model "$MODEL_PATH" --size 512x512 --steps 10 --out-dir results/live-images
# Optional I2I, using a real reference file:
python scripts/verify_sensenova_images.py --base-url http://127.0.0.1:8000 --model "$MODEL_PATH" --image reference.png --size 512x512 --steps 10 --out-dir results/live-images-i2i
```

GPU model path is deployment-dependent; internal NPU commonly uses
/nas/disk1/SenseNova/SenseNova-U1.5-8B-MoT. This script validates image count,
PNG and dimensions, saves images without redundant base64, and explicitly
marks Chat/usage acceptance NOT_COVERED. It does not judge image quality.

## Live metrics (GPU or NPU)

The service must be running with --enable-metrics. These scripts are HTTP
clients; device selection, TP and memory settings belong to the service.

```bash
python scripts/verify_model_metrics.py --api images --base-url http://127.0.0.1:8000 --model "$MODEL_PATH" --size 512x512 --steps 10 --out-dir results/live-image-metrics
python scripts/verify_model_metrics.py --api chat --base-url http://127.0.0.1:8000 --model "$AR_MODEL" --out-dir results/live-chat-metrics
```

Use --metrics-model if model_name differs from the API model argument.
Images can additionally take --image/--n. Evidence includes before.prom,
after.prom, requests.log, summary.json and nested response/image artifacts.
Checks cover positive completed-request/E2E deltas, applicable token or image
counters and nonnegative reported running/waiting gauges. Chat asks for a
multi-token sentence; absent ITL observations fail rather than claiming coverage.
Existing concurrent traffic can affect deltas: this is not an isolated exact
count/performance test. Queue/running transitions, failure/cancellation cleanup,
worker restart, TP aggregation and metrics-on/off overhead still require
separate deployment acceptance.

## Results and archive

```bash
cat results/live-image-metrics/summary.json
cat results/live-chat-metrics/summary.json
tar -czf omni-acceptance-results.tar.gz results/
```

For offline NPU, copy the entire intended checkout including scripts, switch
to that directory and use the same commands; Git/network are not required.
Keep results outside the checkout if packaging code separately.
