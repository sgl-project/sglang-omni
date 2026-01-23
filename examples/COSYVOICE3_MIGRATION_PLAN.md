# CosyVoice3 Self-contained Migration Plan

实现 sglang-omni 自包含支持 CosyVoice3 的实施计划 (Roadmap)。目标是将 CosyVoice3 的核心能力内化到 sglang-omni 中，使其成为原生支持的模型，移除对外部 `CosyVoice` 仓库的依赖。

## Phase 1: 基础设施与配置解耦 (Infrastructure & Config)
**目标**：摆脱对 `hyperpyyaml` 和外部 `cosyvoice3.yaml` 的强依赖，建立 sglang-omni 内部的模型定义标准。

1.  **配置系统迁移**
    *   **现状**：依赖 `load_hyperpyyaml` 动态实例化对象。
    *   **计划**：
        *   定义 Python dataclass 配置类 (e.g., `CosyVoiceConfig`)。
        *   将 yaml 中的超参数（如 `d_model`, `n_layer`, `vocab_size`）硬编码或转为 JSON/Config 格式加载。
        *   不再由 yaml 负责实例化对象，改为在代码中显式构建模型。

2.  **项目结构规划**
    *   建议在 `sglang_omni/models/cosyvoice` 下建立模块：
        *   `configuration_cosyvoice.py`
        *   `modeling_cosyvoice_llm.py` (LLM 结构)
        *   `modeling_cosyvoice_flow.py` (Flow 结构)
        *   `modeling_cosyvoice_hift.py` (Vocoder 结构)
        *   `tokenization_cosyvoice.py` (Tokenizer & Frontend)

## Phase 2: 前端与特征提取 (Frontend & Feature Extraction)
**目标**：实现文本到 Token 的转换及音频特征提取，移除 `CosyVoiceFrontEnd`。

1.  **文本处理 (Text Processing)**
    *   **Tokenizer**：复刻文本 Tokenizer 逻辑（通常是 BPE 或查表），实现 `encode/decode` 接口。
    *   **Text Normalization**：如果使用了复杂的 TN (Text Normalization)，需移植或简化为标准正则处理。

2.  **音频特征提取 (Audio Feature Extraction)**
    *   **Speech Tokenizer**：直接集成 `speech_tokenizer_v3.onnx` 的推理逻辑（使用 `onnxruntime` 或 `torch`）。
    *   **Speaker Embedding**：直接集成 `campplus.onnx` 推理逻辑，用于提取说话人向量。
    *   **Mel Spectrogram**：实现标准的 Mel 谱图计算函数（使用 `torchaudio`）。

## Phase 3: 模型架构移植 (Model Architectures)
**目标**：将核心模型结构定义在 sglang-omni 中，使其符合 PyTorch/vLLM 标准。

1.  **LLM 引擎 (CosyVoice3 LLM)**
    *   **现状**：CosyVoice3 基于 Qwen2 修改。
    *   **计划**：
        *   定义 `CosyVoice3ForCausalLM` 类，继承自 vLLM 的模型基类或 HuggingFace PreTrainedModel。
        *   **重点**：处理 Input Embeddings 的特殊拼接逻辑（文本 + 语音 Prompt + 说话人 Embedding）。这是 CosyVoice3 的核心创新点，必须在 Model 的 `forward` 或 `input_preparation` 中原生支持。

2.  **Flow 引擎 (Flow Matching)**
    *   **现状**：`CausalMaskedDiffWithDiT`。
    *   **计划**：
        *   移植 DiT (Diffusion Transformer) 结构：`TransformerBlock`, `Attention`, `FeedForward`。
        *   移植 Flow Matching 的 Conditional 注入机制（Condition on LLM output tokens）。

3.  **Vocoder 引擎 (HiFT Generator)**
    *   **现状**：`CausalHiFTGenerator`。
    *   **计划**：
        *   移植 HiFiGAN 变体结构：`Generator`, `ResBlock` 等。
        *   确保支持流式 (Streaming) 生成接口。

## Phase 4: 推理逻辑重构 (Inference Logic)
**目标**：在 Stage Processor 中直接调用原生模型，移除 `model.inference()` 封装。

1.  **LLM Stage**
    *   使用标准的 vLLM `LLMEngine` 或 `AsyncLLMEngine` 接口。
    *   实现自定义的 `LogitsProcessor` 或 `SamplingParams` 来控制生成停止条件。

2.  **Flow Stage**
    *   实现 ODE Solver (如 Euler method) 的循环逻辑。
    *   支持 KV Cache (如果 Flow 模型是自回归或由 Masked DiT 优化) 或并行的 DiT 推理。

3.  **Vocoder Stage**
    *   实现 Mel -> Waveform 的前向传播。
    *   处理重叠帧 (Overlap-add) 以支持无缝流式输出。

## Phase 5: 清理与集成 (Cleanup & Integration)
**目标**：完全移除第三方依赖。

1.  **依赖清理**
    *   移除 `sys.path.append("../CosyVoice")`。
    *   移除 `requirements.txt` 中的 `hyperpyyaml`, `cosyvoice`。

2.  **权重转换脚本**
    *   提供脚本将原始 CosyVoice 权重 (`llm.pt`, `flow.pt`, `hift.pt`) 转换为 sglang-omni 原生支持的格式 (如 HF `.bin` 或 `.safetensors`)。
