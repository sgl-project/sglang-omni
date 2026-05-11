# 架构设计 (Architecture)

SGLang-Omni 是一个专为 Omni 模型打造的多阶段流水线框架。Omni 模型特指那些具有多模态输入（文本、音频、图像、视频）和多模态输出的模型。本框架将模型的推理过程拆解为各个专门的阶段（stages），这些阶段可以在异构硬件上独立进行扩展和部署。

## 设计原则 (Design Principles)

- **多阶段解耦 (Multi-stage decomposition)**：Omni 模型包含具有不同计算特征的组件（如编码器、解码器、思考器等）。SGLang-Omni 允许每个阶段拥有独立的资源并独立运行。
- **异步优先 (Async-first)**：整个流水线基于 Python 的 `asyncio` 构建，以实现高并发的请求处理和非阻塞的数据传输。
- **可插拔的数据中继 (Pluggable data relay)**：各阶段之间通过抽象的中继（relay）层进行通信，该层支持共享内存 (SHM)、NCCL、Nixl (RDMA) 以及 Mooncake 后端——从而支持从单机到多节点的各种部署方案。
- **配置驱动 (Configuration-driven)**：流水线通过配置对象进行声明，随后被编译为运行时对象，并由 Runner 启动。无需编写繁琐的连接代码。

## 系统概览 (System Overview)

```text
graph TB
    subgraph API["API 层 (API Layer)"]
        OAI["兼容 OpenAI 的服务器<br/>(FastAPI + Uvicorn)"]
        Client["客户端 (Client)"]
    end

    subgraph Orchestration["流水线编排 (Pipeline Orchestration)"]
        Coord["协调器 (Coordinator)"]
        S1["阶段 1 (Stage 1)"]
        S2["阶段 2 (Stage 2)"]
        SN["阶段 N (Stage N)"]
    end

    subgraph Compute["计算层 (Compute Layer)"]
        W1["工作节点 + 执行器 (Workers + Executors)"]
        W2["工作节点 + 执行器 (Workers + Executors)"]
        WN["工作节点 + 执行器 (Workers + Executors)"]
        E1["引擎<br/>(调度器 + 模型运行器)"]
        E2["引擎<br/>(调度器 + 模型运行器)"]
    end

    subgraph Transport["数据传输层 (Data Transport)"]
        ZMQ["ZMQ 控制平面<br/>(PUSH/PULL/SUB)"]
        Relay["Relay 数据平面<br/>(SHM / NCCL / Nixl / Mooncake)"]
    end

    OAI --> Client
    Client --> Coord
    Coord -- "提交 / 完成" --> S1
    S1 -- "数据就绪" --> S2
    S2 -- "数据就绪" --> SN
    S1 --- W1
    S2 --- W2
    SN --- WN
    W1 --- E1
    W2 --- E2
    S1 & S2 & SN --- ZMQ
    S1 & S2 & SN --- Relay
```

## 模块结构 (Module Structure)

```text
sglang_omni/
├── config/         # 流水线的定义、校验与编译
├── pipeline/       # 协调器、阶段、工作节点以及控制平面
├── engines/        # 计算后端 (OmniEngine, 调度器)
├── executors/      # 面向工作节点的接口，用于桥接阶段与引擎
├── models/         # 特定模型的实现代码 (例如 Qwen3-Omni)
├── preprocessing/  # 多模态输入处理 (音频、图像、视频、文本)
├── relay/          # 跨阶段的数据传输后端
├── proto/          # 请求/消息类型定义及序列化
├── client/         # 包装了协调器的高层客户端
└── serve/          # OpenAI 兼容的 REST API 适配器
```

## 核心组件 (Core Components)

### 流水线配置与编译 (Pipeline Config and Compilation)

配置模块 (`sglang_omni/config/`) 提供了一种声明式定义流水线的方法。一个 `PipelineConfig` 指定了：

- **阶段 (Stages)**：每个 `StageConfig` 命名了一个阶段，引用了一个执行器工厂函数，并定义了一个决定结果去向的路由函数 (`get_next`)。
- **输入处理 (Input handling)**：阶段可以使用 `DirectInput`（直接透传）或 `AggregatedInput`（从多个上游阶段扇入，并附带一个合并函数）。
- **中继配置 (Relay configuration)**：后端类型、设备、缓冲区大小以及流量控制信额。
- **端点 (Endpoints)**：ZMQ 端点分配（单机使用 IPC，多节点使用 TCP）。

`compile_pipeline()` 函数将声明式的配置转换为运行时对象：

1. 验证阶段名称、路由和聚合数据源。
2. 可选地应用**阶段融合 (stage fusion)** —— 将连续的阶段合并到一个工作节点中，以避免中继开销。
3. 分配 ZMQ 端点。
4. 从工厂函数实例化执行器 (executors)。
5. 返回一个 `(Coordinator, [Stage, ...])` 元组。

`PipelineRunner` 负责管理生命周期——启动协调器和所有阶段，然后等待完成或错误。

对于受管的 IPC 启动，建议使用 `build_pipeline_runner()`。较底层的 `compile_pipeline()` 辅助函数会拒绝非受管的 IPC 使用。

### 协调器 (Coordinator)

`Coordinator` (`sglang_omni/pipeline/coordinator.py`) 是中央请求管理器。它的职责包括：

- 注册各个阶段及其 ZMQ 控制端点。
- 跟踪每个请求的全生命周期：`PENDING (等待) → RUNNING (运行中) → COMPLETED (完成) / FAILED (失败) / ABORTED (中止)`。
- 将请求提交给入口阶段。
- 接收完成消息并解析每个请求的 future 对象。
- 通过 ZMQ PUB/SUB 向所有阶段广播中止 (abort) 信号。
- 支持阻塞式 (`submit()`) 和流式 (`stream()`) 两种请求模式。

### 阶段 (Stage)

`Stage` (`sglang_omni/pipeline/stage/runtime.py`) 是一个处理单元，它：

- 通过 ZMQ 控制平面接收传入的工作任务。
- 应用输入处理器——要么直接透传数据，要么等待并合并来自多个上游阶段的输入。
- 使用带有粘性亲和性 (sticky affinity) 的轮询机制将工作路由给其工作节点池（来自同一来源的后续请求会分配给同一个工作节点，以保证缓存局部性）。
- 管理一个 Relay 实例用于下游的数据传输。
- 根据路由函数将完成的结果转发给下一个（或多个）阶段。

### 工作节点 (Worker)

每个 `Worker` (`sglang_omni/pipeline/worker/runtime.py`) 都是阶段内的一个无状态请求处理器。它：

- 从其所属的阶段队列中取出工作项。
- 加载输入数据——可以是内联数据，也可以是根据元数据从中继 (Relay) 获取的张量 (tensors)。
- 将执行任务委托给 `Executor`。
- 将结果分发回阶段，以便路由到下游阶段。
- 通过 asyncio 任务处理大量并发的进行中请求。

### 控制平面 (Control Plane)

控制平面 (`sglang_omni/pipeline/control_plane.py`) 使用 ZMQ 套接字进行阶段间消息传递：

| 套接字 (Socket) | 方向 | 用途 |
|--------|-----------|---------|
| PULL   | 接收   | 传入的工作提交 |
| PUSH   | 发送      | 将工作转发到下一阶段 |
| SUB    | 接收   | 接收来自协调器的中止广播 |

消息类型包括 `SubmitMessage`、`DataReadyMessage`、`CompleteMessage`、`AbortMessage`、`StreamMessage` 和 `ShutdownMessage`，所有消息均使用 msgpack 进行序列化。

### 执行器 (Executors)

执行器 (`sglang_omni/executors/`) 充当阶段与计算后端之间的桥梁：

- **`PreprocessingExecutor`**：将纯函数（如分词、输入归一化）包装为执行器接口。
- **`EngineExecutor`**：将 `Engine`（带有批处理和调度功能）适配为执行器接口，使用请求/结果构建器进行负载转换。
- **`FusedExecutor`**：在编译期间发生阶段融合时，将多个执行器按顺序串联起来。

### 引擎 (Engines)

引擎模块 (`sglang_omni/engines/`) 提供了计算后端：

- **`OmniEngine`**：结合了 `Scheduler` (调度器) 和 `ModelRunner` (模型运行器) 的主引擎。其循环流程为：`调度() → 缓存检查 → 执行() → 缓存更新 → 状态更新()`。
- **`Scheduler`**：与模型无关的请求生命周期管理器。管理状态 (`WAITING → RUNNING → FINISHED`)，并委托给可插拔组件处理：`BatchPlanner` (批处理规划器)、`ResourceManager` (资源管理器)、`IterationController` (迭代控制器)。
- **`ModelRunner`**：消费 `SchedulerOutput` 并调用模型前向传播的无状态执行器。

### 中继 (Relay)

中继模块 (`sglang_omni/relay/`) 提供高性能的阶段间数据传输。详细信息请参阅 [中继设计](./relay_design.md)。

| 后端 | 传输方式 | 适用范围 | 使用场景 |
|---------|-----------|-------|----------|
| SHM     | 共享内存 | 单机 | 低开销，任意 GPU 配置 |
| NCCL    | GPU 集合通信 | 多 GPU | 同步的 GPU 到 GPU 传输 |
| Nixl    | RDMA | 多节点 | 高带宽集群部署 |
| Mooncake | 云优化 | 多节点 | 云端环境 |

所有后端都实现了统一的 `Relay` 接口（包含 `put_async()` / `get_async()`），并通过注册表模式进行选择。

## 请求生命周期 (Request Lifecycle)

```text
sequenceDiagram
    participant C as 客户端
    participant Co as 协调器
    participant S1 as 阶段 1
    participant W1 as 工作节点 1
    participant R as Relay
    participant S2 as 阶段 2
    participant W2 as 工作节点 2

    C->>Co: 提交(OmniRequest)
    Co->>S1: SubmitMessage (ZMQ)
    S1->>W1: 路由任务
    W1->>W1: 通过 Executor 执行
    W1->>R: put_async(tensor)
    R-->>W1: 返回元数据
    W1->>S2: DataReadyMessage(元数据) (ZMQ)
    S2->>W2: 路由任务
    W2->>R: get_async(元数据)
    R-->>W2: 返回张量
    W2->>W2: 通过 Executor 执行
    W2->>Co: CompleteMessage (ZMQ)
    Co-->>C: 返回结果
```

1. **客户端** 向 **协调器** 提交一个 `OmniRequest`。
2. 协调器通过 ZMQ 向 **入口阶段** 发送 `SubmitMessage`。
3. 阶段的 **输入处理器** 处理该请求，然后 **路由器** 将其分配给一个工作节点。
4. **工作节点** 通过其 **执行器** 执行请求。
5. 如果存在下游阶段，工作节点会将输出张量写入 **Relay**，并向下个阶段发送包含中继元数据的 `DataReadyMessage`。
6. 下一阶段的工作节点从中继获取张量并继续处理。
7. 最终阶段向协调器发回 `CompleteMessage`。
8. 协调器将结果解析给客户端的 future 对象。

## 数据流模式 (Data Flow Patterns)

### 顺序 (Sequential)

```text
graph LR
    A["阶段 A"] -->|relay| B["阶段 B"] -->|relay| C["阶段 C"]
```

最简单的模式：每个阶段将其输出传递给下一个阶段。

### 扇出 / 分发 (Fan-Out)

```text
graph LR
    A["预处理"] -->|relay| B["图像编码器"]
    A -->|relay| C["音频编码器"]
```

路由函数将输出并行引导至多个下游阶段。当不同模态需要各自独立的编码器时使用此模式。

### 扇入 / 聚合 (Fan-In / Aggregation)

```text
graph LR
    B["图像编码器"] -->|relay| D["聚合阶段"]
    C["音频编码器"] -->|relay| D
```

聚合输入处理器会等待所有上游数据源就绪，然后使用自定义函数将它们合并，最后再传递给工作节点。

### Qwen3-Omni 流水线示例

一个结合了上述模式的具体示例：

```text
graph TD
    PP["预处理<br/>(分词, 模板化)"]
    IE["图像编码器"]
    AE["音频编码器"]
    AGG["聚合<br/>(合并隐变量与 token)"]
    TH["思考器<br/>(自回归解码器)"]
    DEC["解码阶段<br/>(音频合成)"]

    PP -->|"包含图像"| IE
    PP -->|"包含音频"| AE
    IE --> AGG
    AE --> AGG
    AGG --> TH
    TH --> DEC
```

预处理阶段根据输入模态扇出到被激活的编码器。聚合阶段将编码器的输出与文本 token 扇入并合并。思考器运行自回归解码，随后解码阶段合成音频输出。

## 配置示例 (Configuration Example)

```python
from sglang_omni.config import (
    ExecutorConfig, PipelineConfig, StageConfig,
    build_pipeline_runner,
)

config = PipelineConfig(
    name="my_pipeline",
    entry_stage="preprocess",
    stages=[
        StageConfig(
            name="preprocess",
            executor=ExecutorConfig(
                factory="my_project.executors.create_preprocess",
                args={},
            ),
            get_next="my_project.routing.preprocess_next",
        ),
        StageConfig(
            name="thinker",
            executor=ExecutorConfig(
                factory="my_project.executors.create_thinker",
                args={"model_path": "Qwen/Qwen3-Omni"},
            ),
            get_next="my_project.routing.end",
        ),
    ],
)

runner = build_pipeline_runner(config)
```

## 通信分层 (Communication Layers)

| 层级 | 技术 | 用途 |
|-------|-----------|---------|
| 控制平面 | ZMQ (PUSH/PULL/SUB) | 阶段间任务提交，中止信号广播 |
| 数据平面 | SHM / NCCL / Nixl / Mooncake | 阶段间的张量传输 |
| 请求跟踪 | 内存字典 + asyncio futures | 协调器维护请求生命周期 |
| 工作节点调度 | asyncio + 队列 | 阶段内并发请求的分发 |
| 外部 API | FastAPI + Uvicorn | 兼容 OpenAI 的 HTTP 端点 |

## 关键设计决策 (Key Design Decisions)

**为什么控制平面使用 ZMQ？**
ZMQ 提供了轻量级的无代理消息传递机制，并具备灵活的套接字模式（用于工作分发的 PUSH/PULL，用于广播的 PUB/SUB）。它避免了消息代理服务器的开销，同时支持 IPC 和 TCP 传输。

**为什么要分离控制平面和数据平面？**
控制消息（提交、完成、中止）体积小且对延迟敏感。而张量数据体积大且对吞吐量敏感。将它们分离允许各自使用最优的传输方式——ZMQ 用于控制，RDMA/NCCL 用于数据。

**为什么要使用配置驱动的流水线？**
声明式配置使得流水线具备可复现性、版本控制能力且易于修改。编译步骤会自动处理所有的连接建立、端点分配以及优化（如阶段融合）。

**为什么要使用粘性工作节点亲和性 (sticky worker affinity)？**
当一个请求返回到同一个阶段时（例如进行迭代解码），将其路由到同一个工作节点可以保持 KV Cache（键值缓存）的局部性，从而避免冗余的重新计算。
