# Response and Metrics acceptance

The response scope is Images-only. Chat and /generate retain their existing
HTTP contracts; metrics do not depend on newly added HTTP usage fields.

## Model-free regression (Linux)

Use the intended Python environment from the checkout under test:

```bash
python scripts/verify_response_usage.py --repo "$PWD" --mode smoke --out-dir results/response-smoke
python scripts/verify_response_usage.py --repo "$PWD" --mode full --out-dir results/response-full
python scripts/verify_metrics.py --repo "$PWD" --mode smoke --out-dir results/metrics-smoke
python scripts/verify_metrics.py --repo "$PWD" --mode full --out-dir results/metrics-full
```

The main-based branch tests common image response utilities; the SenseNova
acceptance branch also tests its Images API and model token accounting.
Full response regression includes existing audio compatibility tests. Record
baseline/environment failures rather than silently skipping them.
These checks do not load models or establish GPU/NPU correctness.

## Live SenseNova Images response and metrics

Use test/omni-response-usage-sensenova for combined metrics/Images acceptance.
The main-based metrics branch does not include SenseNova or Images routes.
Start the model service with --enable-metrics for the commands below:

```bash
python scripts/verify_model_metrics.py --api images --base-url "$BASE_URL" --model "$MODEL_PATH" --size 512x512 --steps 10 --out-dir results/metrics-images
python scripts/verify_model_metrics.py --api images --base-url "$BASE_URL" --model "$MODEL_PATH" --image reference.png --size 512x512 --steps 50 --n 2 --out-dir results/metrics-i2i
```

The image client validates usage, output format, dimensions and image counts.
The metrics client saves before/after snapshots and checks positive completed
request, E2E histogram and generated-image deltas. It does not treat image input
usage as AR prompt/generation token metrics or validate image-editing quality.
Results include summary.json, before.prom, after.prom, requests.log and responses/.

## Live AR metrics

```bash
python scripts/verify_model_metrics.py --api chat --base-url "$BASE_URL" --model "$AR_MODEL" --out-dir results/metrics-chat
```

The Chat client uses non-streaming and default streaming requests without
stream_options. Existing streaming usage may appear on the finish chunk; this
is compatibility checking, not new Chat usage adaptation. Cache details are
internal metrics input and are not added to Chat or /generate responses.

## Acceptance boundaries

Use an idle, dedicated service for finished-request deltas. Concurrent traffic
can contribute to snapshots. This does not establish queue/running transitions
under load, abort/failure recovery, worker restarts, TP aggregation correctness,
or monitoring overhead. These require separate hardware/load acceptance.
Do not commit generated results.
