# Prometheus Metrics

SGLang-Omni can expose an optional Prometheus-compatible `/metrics` endpoint
for the Omni API/coordinator layer.

Start the server with:

```bash
sgl-omni serve ... --enable-metrics
```

Scrape:

```bash
curl http://localhost:8000/metrics
```

This endpoint currently exposes low-cardinality Omni API/coordinator metrics
only. It does not enable, proxy, or aggregate underlying SGLang stage-level
metrics.

The endpoint intentionally does not expose request IDs, prompts, file paths,
voice names, or arbitrary user input as Prometheus labels. HTTP request metrics
also omit `model_name`; an Omni API server process serves one configured model,
and Prometheus target labels such as `job` and `instance` should identify the
scrape target.

The HTTP duration histogram measures request handler duration. For streaming
responses, it does not represent full end-to-end stream duration.

For GPU and system metrics, use infrastructure exporters such as DCGM exporter,
node exporter, cAdvisor, or kube-state-metrics.
