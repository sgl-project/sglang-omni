---
name: sglang-omni-progress-and-reference-audio-pr
overview: 为 sglang-omni 创建本地中文 PROGRESS.md（不提交）追踪全局进展，并基于主旨（OpenAI 兼容语音 serving）提交 1 个高质量、不与上游重复、且可在 1-2 张 A100 40G 上验证的 onboarding PR（方向：把 reference-audio / voice-cloning 能力通过 OpenAI 兼容接口暴露出来，或在确认已支持后改为补全某单卡可跑小模型的端到端 example+config+文档）。
todos:
  - id: write-progress-md
    content: 创建根目录中文 PROGRESS.md，记录架构、进展、缺口与本轮贡献点（仅本地，不提交）
    status: completed
  - id: add-example-yamls
    content: 新增 examples/configs/{higgs_tts,zonos2,fun_asr}.yaml，使用正确 config_cls 与官方 model_path
    status: completed
  - id: update-cookbooks
    content: 在 docs/cookbook 的三个 md 末尾追加 Quick Start/Serving example 段落引用 yaml
    status: completed
    dependencies:
      - add-example-yamls
  - id: add-smoke-test
    content: 为 ZonOS2 或 Fun-ASR 补最小单卡端到端 serving smoke 测试，复用 test_model 模式
    status: completed
    dependencies:
      - add-example-yamls
  - id: update-examples-readme
    content: 更新 examples/README.md 索引，登记三个新 example 与单卡说明
    status: completed
    dependencies:
      - add-example-yamls
      - update-cookbooks
  - id: sync-and-open-pr
    content: 同步 fork 与上游，切分支提交并开 PR，描述单卡验证与无重复声明
    status: completed
    dependencies:
      - update-cookbooks
      - add-smoke-test
      - update-examples-readme
---

## 用户需求

以专业开源贡献者身份，为 sglang-omni 项目（统一多模态/语音推理 serving runtime，核心为 OpenAI 兼容语音 serving）维护一份全局中文进展文档，并基于项目主旨提交 1 个高质量、不与上游重复的 Pull Request。

## 两项产出

1. **本地中文 PROGRESS.md**：位于仓库根目录，全中文，记录整体架构、已完成功能、待解决问题、未来规划与本轮贡献点；仅用于本地审查，不纳入 git/远程。
2. **1 个高质量 onboarding 完善 PR**：在 fork 仓库（1571859588/sglang-omni）开分支，提向 sgl-project/sglang-omni 的 main 分支。

## 核心约束

- 硬件仅 1-2 张 A100 40G 可用，排除所有需多卡/大显存模型（Qwen3-Omni、MOSS 大模型、Ming-Omni 等）。
- 不与现有上游 PR 重复（现有 PR 集中在 Qwen3-Omni 性能优化、MOSS-TTS 采样融合、消费级 GPU 支持、TTS 修复等）。
- 经探查确认：serving 层 reference-audio/voice cloning 已完整支持，故 onboarding 方向转为「单卡可跑小模型的端到端示例与 smoke 测试补全」。

## 选定 PR 范围（已验证为真实缺口）

补全以下单卡可跑小模型的端到端 onboarding 资产（config 类已存在、cookbook 已存在但缺 example yaml 与可执行引用）：

- Higgs-4B TTS（config 类 HiggsTtsPipelineConfig，model_path bosonai/higgs-audio-v3-tts-4b）
- ZonOS2（config 类 Zonos2PipelineConfig，单卡 colocated 默认）
- Fun-ASR（config 类 FunASRPipelineConfig，默认 cuda:0，Nano 极小显存）

## 技术栈

- 语言/框架：Python 3.11+，复用 sglang-omni 现有架构（models 注册机制、PipelineConfig/StageConfig 声明式配置、serve 层 OpenAI 兼容接口、pytest 测试体系）。
- 配置格式：YAML（examples/configs/*.yaml，与现有 18 个 yaml 同构）。
- 测试：复用 tests/test_model 既有端到端 serving CI 模式（单卡 colocated server + 轻量断言），不引入新测试框架。

## 实现方案

通过补全「端到端可用示例 + 文档引用 + 单卡 smoke 验证」三类资产，使单卡可跑的小模型 onboarding 从「代码可用但无开箱示例」升级为「文档可查、配置可直接 serve、CI 可验证」，契合项目「OpenAI 兼容语音 serving」主旨且不与现有 PR 冲突。

关键技术决策：

1. **example yaml 复用现有 config 类**：直接写 `config_cls: HiggsTtsPipelineConfig` / `Zonos2PipelineConfig` / `FunASRPipelineConfig` + 官方 `model_path`，与 Voxtral/S2Pro 的 yaml 完全一致，零新增抽象，避免架构漂移。
2. **单卡显存友好默认**：yaml 仅声明 config_cls 与 model_path，运行时显存预算由模型 config 的 `process_safe_edges`/colocated 默认值决定（ZonOS2 默认即单卡 colocated；Fun-ASR 默认 gpu=0；Higgs-4B 单卡可服务），无需额外调参。
3. **smoke 测试只补最干净的一个模型**：优先 ZonOS2 或 Fun-ASR（单卡、依赖最小），以最小端到端 serving 断言（启动 server → 一次请求 → 非空音频/文本）补强，复用 tests/test_model 现有 fixture 模式；Higgs 已有 test_tts_serving_ci.py 单卡 CI，不重复造轮。
4. **cookbook 增量更新**：仅在现有 cookbook 末尾追加「Quick Start / Serve with example config」段落，引用新增 yaml，不改写既有评测/架构内容，控制 blast radius。

## 性能与可靠性

- 所有改动为配置/文档/测试，无运行时热路径变更；yaml 与 cookbook 不改变模型执行逻辑。
- smoke 测试显式标记单卡需求（如 pytest marker 或 docstring 声明 CUDA_VISIBLE_DEVICES），避免误占多卡 CI。

## 执行注意

- 提交前先在本机将 fork 与上游 main 同步（git remote 现状执行阶段确认），再切分支、commit、push、开 PR。
- PR 描述需声明：单卡验证方式、与现有 PR 无重叠、关联 issue（若上游有对应 onboarding issue 则引用）。
- PROGRESS.md 仅落盘根目录，绝不执行 git add/commit。

## 目录结构（将创建/修改的文件）

```
/mnt/public/nyt1/infra/projects/sglang-omni/
├── PROGRESS.md                         # [NEW, 仅本地] 中文全局进展文档，不提交
├── examples/
│   ├── configs/
│   │   ├── higgs_tts.yaml              # [NEW] config_cls: HiggsTtsPipelineConfig + model_path
│   │   ├── zonos2.yaml                 # [NEW] config_cls: Zonos2PipelineConfig + model_path
│   │   └── fun_asr.yaml                # [NEW] config_cls: FunASRPipelineConfig + model_path
│   └── README.md                       # [MODIFY] 在索引表增加三个新 example 的条目与说明
├── docs/cookbook/
│   ├── higgs_tts.md                    # [MODIFY] 末尾追加 Quick Start/Serving example 段落
│   ├── zonos2.md                       # [MODIFY] 末尾追加 Quick Start/Serving example 段落
│   └── fun_asr.md                      # [MODIFY] 末尾追加 Quick Start/Serving example 段落
└── tests/test_model/
    └── test_zonos2_or_fun_asr_smoke_ci.py  # [NEW] 单卡端到端 serving smoke 测试（最小断言）
```