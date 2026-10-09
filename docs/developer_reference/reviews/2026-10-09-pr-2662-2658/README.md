# PR #2662 / #2658 GPU 验证

## 范围与判据

验证 MOSS reference encode 共享实现和 Qwen3-TTS 首块 tail decode 的正确性。使用原 PR 的测试及原有断言，分别记录 CPU 与 accelerator 用例；检查每项 skip。失败须在相同环境下与 base 比较后，才能归因于 PR。本验证不复现端到端吞吐、WER 或 speaker similarity。

## Source of Truth

- [PR #2662](https://github.com/sgl-project/sglang-omni/pull/2662)：9855cdd76c00290e738304f4d9d7755a45f992a8。
- [PR #2658](https://github.com/sgl-project/sglang-omni/pull/2658)：e91214430bb6dc8997bfde192399094620def779。
- 精确测试列表、marker 和归档 SHA256 见 config.json 与 provenance.json。
- 环境采用 PR CI 镜像 digest，实际包版本和 GPU 型号由 runner.py 写入 environment.json。
- 原始 JUnit、日志和结果摘要保存在本目录；不产生可复用数据集或 checkpoint。

## 运行

Radix 单卡 H200，租约 01M4GY0J1PWD660J9BMVCHB9W8，hyper01 GPU 0，2026-10-09 18:15–20:15 UTC。只有本任务授予的卡可见。节点数据仅作为当次租约暂存；结束前回收日志并提交 Git。

将两个固定 SHA 的源码分别解压至 code/00 与 code/01，将 runner.py 复制为 main.py 并与 config.json 放在同一目录；运行 python main.py。每组测试使用独立 Python 进程和对应源码路径，CPU/GPU 结果分开统计。

## 当前状态

源码归档已在节点下载，SHA256 与本机从 GitHub 获得的归档一致。GPU 发射前五秒检查占用为零。镜像拉取中，尚无 GPU 测试结果。
