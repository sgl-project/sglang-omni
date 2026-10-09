# PR #2662 / #2658 GPU 验证

两个固定 PR head 的针对性回归全部通过：633 项 CPU、46 项 GPU，0 failure、0 error、0 skip。未发现需要新增的代码阻塞意见。

| PR | Head | CPU | GPU |
| --- | --- | ---: | ---: |
| [#2662](https://github.com/sgl-project/sglang-omni/pull/2662) | 9855cdd76c00290e738304f4d9d7755a45f992a8 | 268 | 22 |
| [#2658](https://github.com/sgl-project/sglang-omni/pull/2658) | e91214430bb6dc8997bfde192399094620def779 | 365 | 24 |

## 验证范围与边界

MOSS 覆盖共享 reference encode、缓存、重复请求合并、异常隔离、worker 生命周期、CUDA stream，以及相关 encoder/tokenizer 和两个模型的 pipeline。Qwen 覆盖增量 codec、tail 感受野、不同参考长度的批处理、音频与后续状态一致性、CUDA graph replay、共享 pool 和 streaming scheduler。

使用原 PR 测试与原有断言，未更改生产代码或测试容差。CPU 与 accelerator 用例分开运行。benchmark 标记用例未执行，其中包括需要真实 Qwen tokenizer checkpoint 的独立 gate；本结果不证明真实 checkpoint 的全场景音质等价，也不复现 PR 的端到端吞吐、WER 或 speaker similarity 指标。

## 环境

- Radix hyper01，单张 NVIDIA H200；只注入获批 GPU 0。
- Python 3.12.3；PyTorch 2.13.0+cu130；CUDA 13.0；SGLang 0.5.21；Transformers 5.12.1；qwen-tts 0.1.1；Triton 3.7.1。
- 使用 PR CI 的固定镜像 digest，在 /data/runtime 建独立环境，将镜像自带 SGLang 0.5.19 更新到 PR 声明的 0.5.21。完整版本见 results/packages.json 和 results/environment.json。
- GPU 发射前五秒检查显存与利用率均为零。四组 pytest 均正常退出，约 134 秒测试时间。

## Source of Truth

- 本目录的 config.json 是精确用例和 marker 列表；provenance.json 保存 PR SHA、源码归档 SHA256、镜像 digest 和验证判据。
- results/ 保存四组原始日志、JUnit XML、环境和执行命令。回收归档 SHA256：98d0e7c2e2802efdf78c434ce32b77c5792e0e9e239ead4e4521c18d7d3a76e0；11 个结果文件已逐个核对哈希。
- runtime.json 保存租约、容器、GPU UUID 与节点临时路径的对应关系。节点存储仅用于本次运行；Git 是保留结果的正本。
- 不产生可复用数据集、模型参数或 checkpoint。

## 复现

将两个固定 SHA 的源码分别解压至 code/00 与 code/01，将 runner.py 复制为 main.py，与 config.json 放在同一目录。准备上述依赖并只暴露一张 CUDA GPU 后，执行 python main.py。每组测试由独立 Python 进程从对应源码目录运行；输出写入 outputs/。

## 资源状态

测试已结束，结果已回收并提交 Git。任务容器与 map 记录已删除；Radix 确认租约已释放，账户无活跃分配。释放回执见 cleanup.json。
