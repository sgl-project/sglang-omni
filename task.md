# 任务：完成 SGLang-Omni 中 dots.tts 的 OmniScheduler 迁移

## 仓库与路径

- 工作区：`/data/chenyang/sglang-omni` 在 `dots-tts` branch 下。
- 权重：`/data/chenyang/models/dots.tts-mf`
  - HF architecture：`DotsTTSForConditionalGeneration`
  - 文件：`config.json`、`llm_config.json`（Qwen2）、`model.safetensors`、
    `vocoder.safetensors`、`speaker_encoder.safetensors`、`latent_stats.pt`、tokenizer
- 上游参考仓库（可选）：`/tmp/dots.tts`
- 计划文件（部分内容可能过时）：`/root/.cursor/plans/dots_omnischeduler_migration_efb3a65a.plan.md`
- Python 环境：`/data/chenyang/.python/omni`（`source .../bin/activate` 或 `ca omni`），读取 `/root/.zshrc`。
- CUDA 环境注意：用户 zshrc 会把 host `libcuda`（`/lib/x86_64-linux-gnu`）提前；但对
  torch 2.11+cu130，必须把 nvidia wheel 库（尤其
  `site-packages/nvidia/cudnn/lib`）放在 `/lib/x86_64-linux-gnu` **之前**，否则系统
  cuDNN 9.10 会盖住 torch 自带的 9.19 并崩溃。

GPU 工作前推荐环境：

```bash
source /data/chenyang/.python/omni/bin/activate
export LD_LIBRARY_PATH="$(python - <<'PY'
from pathlib import Path
site = Path('/data/chenyang/.python/omni/lib/python3.12/site-packages')
libs = ['nvidia/cudnn/lib','nvidia/cublas/lib','nvidia/cuda_runtime/lib',
        'nvidia/cuda_nvrtc/lib','nvidia/cufft/lib','nvidia/curand/lib',
        'nvidia/cusolver/lib','nvidia/cusparse/lib','nvidia/nccl/lib',
        'nvidia/nvjitlink/lib','torch/lib']
print(':'.join([str(site/n) for n in libs if (site/n).is_dir()] + ['/lib/x86_64-linux-gnu']))
PY
)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 只选一张空闲卡，例如 CUDA_VISIBLE_DEVICES=1
```

## 产品目标

在 SGLang Omni 中接入小红书 **dots.tts-mf** serving：

- 首版范围：**仅 mf**、非流式、零样本 **continuation cloning**
  （恰好一段本地参考 wav + 非空参考文本）
- 输出 48 kHz（AudioVAE vocoder）
- 质量与吞吐门禁：Seed-TTS-Eval **EN 1088**，客户端 **concurrency=16**：
  - **吞吐：** 1088 条 TTS 请求必须在 **10 分钟内**跑完（否则几乎可以断定仍在用
    native Torch AR，而不是 SGLang backend）。
  - **质量：** corpus WER **必须 < 3%**；官方参考约 **1.29%**（NFE=4）。≥3% 视为未达标。
  - 若 WER 远高于官方，通常是 feedback / latent / VAE 路径坏了。

## 硬性架构要求（不可妥协）

**必须使用 SGLang backend（OmniScheduler + SGLang KV / batching）。**
**严禁使用 native Torch backend 做 AR serving。**

此前 PR1 用了 Audar 风格的 `SimpleScheduler` + in-tree
`DotsTtsModel.generate_latents()`，LLM KV 走 HuggingFace `Qwen2ForCausalLM` +
`StaticCache`（`components/backbone/llm_inference.py`）。这就是被禁止的
**native Torch AR backend**：在 concurrency=16 下也拉不出合理吞吐，全量 EN
会慢到数十分钟乃至更久——这在本任务中不可接受。

用户要求：

1. **`tts_engine` 只能走 OmniScheduler + SGLang backend**（SGLang 管 LLM KV /
   continuous batching / 调度）。禁止 HuggingFace `generate`、禁止自管
   `StaticCache` / `DynamicCache` 的 PyTorch AR 循环、禁止
   `SimpleScheduler(_generate)` 包一层 `generate_latents` 冒充 serving。
2. **禁止留下任何 native Torch AR 代码路径**、fallback、“临时还能跑”的双实现。
   迁完后仓库里不得再有可调用的 Torch AR serving 入口。
3. 代码尽量 **简洁**；一条路径做完即可。
4. TTS **单卡**即可（所有 stage `gpu: 0`，对应一个 `CUDA_VISIBLE_DEVICES`）。
5. 非 AR 阶段可以继续用 `SimpleScheduler`（preprocessing / reference_encode /
   audio_decode）——这是仓库惯例，**不等于**允许 Torch AR backend。只有
   `tts_engine` 必须走 SGLang Omni。

**正确对标：** Ming-Omni-TTS（连续 latent + CFM/DiT + OmniScheduler / SGLang），
**不是** Voxtral/Higgs 离散 codec，更不是 Audar 式 llama.cpp / HF StaticCache 自循环。

优先阅读：

- `sglang_omni/models/ming_tts/engine_builder.py`
- `sglang_omni/models/ming_tts/model_runner.py`
- `sglang_omni/models/ming_tts/sglang_model.py`（`run_tail_step`）
- `sglang_omni/models/ming_tts/stages.py`（`create_sglang_tts_engine_executor`）
- `sglang_omni/models/ming_tts/engine_io.py`
- `sglang_omni/models/ming_tts/hf_config.py`
- `sglang_omni/models/ming_tts/weight_loading.py`
- `sglang_omni/model_runner/sglang_model_runner.py` 的 `_register_omni_model`
- `sglang_omni/model_runner/model_worker.py`（arch → llm_config）
- `docs/developer_reference/tts_model_integration.md`

## 当前进度（交接时状态）

Git：`main`，相对 origin ahead 5；大量 dots 相关改动 **尚未提交**。

### 已完成 / 可复用

四阶段 glue：

- `sglang_omni/models/dots_tts/{config,request_builders,payload_types,prompt_builder,reference_encode,audio_decode,__init__}.py`
- `examples/configs/dots_tts.yaml`
- `docs/cookbook/dots_tts.md`（以及 `docs/basic_usage/tts.md`、`tts_process_topology.md` 中的表格行）
- `pyproject.toml` 已加 `torchdiffeq>=0.2.5`
- 单测：`tests/unit_test/dots_tts/`（config / request_builders / reference_encode）
- in-tree 推理组件：`sglang_omni/models/dots_tts/components/`（DiT、vocoder、speaker、
  schedule/tokenizing 等，大体从上游 dots.tts 移植）

### Omni 脚手架（已有文件，但尚未真正接线）

下列文件已开始写，但 serving 仍走 SimpleScheduler：

- `hf_config.py` — `DOTS_TTS_MODEL_ARCH_OVERRIDE = "DotsTTSSGLangModel"`，`DotsTTSConfig`
- `sglang_model.py` — `DotsTTSSGLangModel`，含 `run_tail_step`、Qwen2 SGLang backbone、侧车模块
- `weight_loading.py`
- `engine_io.py` — `DotsTtsSGLangRequestData`、`make_dots_tts_scheduler_adapters`

### 仍缺失 / 仍错误

- **缺失：** `engine_builder.py`、`model_runner.py`
- **`stages.py` 仍在用** `SimpleScheduler` + `DotsTtsModel.generate_latents()`（Torch AR）
- **yaml** 仍带旧路径 knobs（如 `optimize: false`、ode 参数挂在 SimpleScheduler engine 上）
- **未注册：** `sglang_model_runner.py` 的 `_register_omni_model` 里没有 `DotsTTSSGLangModel`
- **Torch AR 死路径仍在：**
  - `components/model.py` 的 `generate_latents` / `generate_audio` / `_generate_latents_stream`
  - `components/backbone/llm_inference.py`（HF StaticCache）
  - core/model 里若仅为该 AR 服务的 HF `Qwen2ForCausalLM` 路径
- 尚无 Omni 侧的 engine_io / builder / runner 合同单测
- Omni 路径上的完整 Seed-TTS EN 门禁未完成
  - 旧 Torch 路径曾 smoke 成功；8 条样本 WER 约 1.09% 看起来正常
  - 全量 EN 在 concurrency=1 下极慢，已被判定不合理并停止
  - 用户要求走 Omni，评测用默认 concurrency **16**

## 目标架构

```
preprocessing (SimpleScheduler)
  -> reference_encode (SimpleScheduler；加载/缓存参考波形并构建 schedule)
  -> tts_engine (OmniScheduler + SGLang Qwen2 KV)
  -> audio_decode (BatchVocoder / AudioVAE；SimpleScheduler 外壳)
```

单步 decode（对齐 Ming）：

1. `before_decode`：把上一 patch 的 feedback embed 写入本步
2. SGLang Qwen2 backbone forward → hidden
3. `post_decode` / `run_tail_step`：MeanFlow DiT 采样 latent patch 并 append；
   `eos_proj` 对比 `eos_threshold`；否则 `patch_encoder(latent)` 作为下一 feedback
4. Prefill：`generation_schedule`，prompt 音频 span 用 prompt latents 填充

首版 server args 约束（与 Ming 一致）：

- `disable_radix_cache=True`
- `chunked_prefill_size=0`
- 单卡；不要发明多卡拓扑

## 具体要完成的目标

1. 完成 Omni 接线：
   - 实现 `DotsTtsEngineBuilder` + `DotsTTSModelRunner`
   - 将 `create_tts_engine_executor` 改为只走 `builder.build()`（可保留 Ming 风格别名）
   - 在 `_register_omni_model` 注册 `DotsTTSSGLangModel`（必要时改 `model_worker.py`）
   - 更新 `config.py` / `examples/configs/dots_tts.yaml` 的 Omni factory 参数
2. **彻底删除** Torch AR serving 路径（禁止双路径）：
   - 删除 SimpleScheduler + `generate_latents` engine
   - 删除迁移后不再使用的 `llm_inference.py` / `generate_*` AR 闭环
   - 保留 `run_tail_step` 与 `audio_decode` 仍需要的 DiT / vocoder / speaker / schedule 工具
3. 更新 docs/cookbook：写明 OmniScheduler；Seed-TTS 使用 **默认 concurrency 16**
   （不要把 concurrency=1 写成评测配方）
4. 测试：CPU 单测覆盖 adapters/config/request 合同；能不加 GPU 的地方补 Ming 风格
   runner/engine_io 测试
5. 验证门禁（两项同时满足才算过）：
   - 单卡 `sgl-omni serve --config examples/configs/dots_tts.yaml --allowed-local-media-path /`
   - HTTP `/v1/audio/speech` smoke → 合法 wav
   - `python -m benchmarks.eval.benchmark_tts_seedtts` 跑满 EN **1088**，
     **`--concurrency 16`**（或默认 16），再 ASR 算 WER
   - **吞吐硬门槛：** concurrency=16 时，1088 条 TTS **必须在 10 分钟内结束**；
     超时说明仍在用 native Torch backend 或 Omni 未真正批处理，任务失败
   - **质量硬门槛：** corpus WER **< 3%**（目标贴近官方 1.29%）；≥3% 未达标

推荐评测流程（benchmark 自带的 `managed_omni_server` **不会**传 `--config`；dots 请自行起服）：

```bash
# 终端 A — TTS，1 张 GPU
sgl-omni serve \
  --model-path /data/chenyang/models/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --allowed-local-media-path / \
  --host 127.0.0.1 --port 18123

# 终端 B — 生成
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model dots-tts-mf \
  --base-url http://127.0.0.1:18123 \
  --use-existing-server --generate-only \
  --lang en --concurrency 16 \
  --output-dir results/dots_tts_seedtts_en

# 终端 C — ASR WER，另选一张空闲 GPU
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model dots-tts-mf \
  --transcribe-only --lang en \
  --port 18124 \
  --output-dir results/dots_tts_seedtts_en
```

## 完成条件（Definition of Done）

- [ ] `tts_engine` **只走 SGLang backend（OmniScheduler）**；仓库中不再存在
      native Torch AR（SimpleScheduler + `generate_latents` / HF StaticCache /
      `LLMInference` 自循环）任何可调用路径
- [ ] `DotsTTSSGLangModel` 已注册，并能加载 `/data/chenyang/models/dots.tts-mf`
- [ ] 单卡 serve + `/v1/audio/speech` smoke 能产出合法 48 kHz 音频
- [ ] `pytest tests/unit_test/dots_tts -q` 通过
- [ ] Seed-TTS EN **1088 @ concurrency 16**：TTS 生成在 **10 分钟内完成**
- [ ] 同上设置下 corpus WER **< 3%**（目标贴近官方 1.29%）
- [ ] cookbook/docs 明确写 SGLang / OmniScheduler，评测 concurrency=16，并写明
      10 分钟 / WER < 3% 门禁
- [ ] 代码保持精简，无 Torch/SGLang 双实现

## 风格与流程注意

- SPDX Apache-2.0 头；类型注解；早返回；不要中文注释
- 非显然注释只用：`# note (chenyang): ...`
- 遵循仓库 TTS 集成惯例；优先对齐 Ming，不要另起一套抽象
- 除非用户明确要求，否则不要 commit / push
- 本轮不要开 radix cache / chunked-prefill rollback
- 不要强行把 preprocess / vocoder 改成 OmniScheduler
- `torchdiffeq` 已在 pyproject；若 omni venv 缺包则安装：
  `uv pip install 'torchdiffeq>=0.2.5' --python /data/chenyang/.python/omni/bin/python`

## 已知坑

1. 只把 `/lib/x86_64-linux-gnu` 放最前会导致 cuDNN 与 torch 2.11 冲突——nvidia wheel 库要更靠前
2. `benchmark_tts_seedtts` 默认 concurrency 是 **16**。必须用 16 验收：正确的
   SGLang backend 下，1088 条应在 **10 分钟内**结束。若仍要数十分钟，说明还在
   native Torch AR——立刻停掉去修 backend，不要用 concurrency=1“硬扛完”。
3. benchmark 的 `managed_omni_server` 不传 `--config`；dots 请自行 `serve`，再用
   `--use-existing-server --generate-only`
4. tokenizer 可能报 mistral regex 警告；上游 dots 同样加载方式——除非 WER 很差，不要盲目“修”
5. prompt audio 必须放到 engine device（曾有 bug：需要 `.to(device=..., dtype=float32)`）
6. audio payload 键是 `audio_waveform` bytes + shape/dtype，不是顶层 `audio` 数组

## 本轮不做（后续）

- StreamingVocoder / 流式输出
- soar 权重、x-vector-only / text-only 模式
- 把 DiT CUDA Graph / torch.compile 当主路径
- 带连续 embed 内容哈希的 radix cache
- 多卡拆分

## 建议开工顺序

1. 阅读 Ming TTS Omni 文件 + 现有 dots `sglang_model.py` / `engine_io.py`
2. 补齐 `engine_builder.py` + `model_runner.py`
3. 重写 `stages.py` + yaml + registry 注册
4. 删除 Torch AR 死代码
5. 补单测
6. 单卡 smoke
7. Seed-TTS EN 1088 @ concurrency 16 + WER 门禁
