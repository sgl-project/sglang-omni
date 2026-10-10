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
include the service-level `model_name` label so Kubernetes deployments with many
Omni API server replicas can aggregate request rates and latency by model.

The HTTP duration histogram measures request handler duration. For streaming
responses, it does not represent full end-to-end stream duration.

For GPU and system metrics, use infrastructure exporters such as DCGM exporter,
node exporter, cAdvisor, or kube-state-metrics.
