# Relay 基准测试 (Benchmark Relay)

Relay（中继）是 SGLang-Omni 的核心组件。它负责在不同处理阶段（stages）之间传输数据。我们提供了一个基准测试脚本，用于测量不同通信后端的性能表现。

## 基准测试脚本

```bash
python benchmark_relay.py \
    --backend-type all \
    --start-size 16 \
    --end-size 1024 \
    --factor 2 \
    --output-dir ./results
```
