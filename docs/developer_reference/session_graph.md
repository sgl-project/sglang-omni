# Session Graph

A session graph runs several models (or tools) as one long-lived session. Each
**node** is an ordinary session over a linear stage chain; the graph joins
nodes with **edges** and is launched with the rest of the pipeline from one
config.

```python
GraphConfig(
    nodes={
        "voicechat": GraphNodeConfig(stages=["perception", "thinker", "talker", "code2wav"]),
        "tool": GraphNodeConfig(stages=["tool"]),
    },
    inputs={"audio": ["voicechat"]},   # client input modality -> nodes
    output="voicechat",                # node whose data returns to the client
    edges=[
        GraphEdgeConfig(source="voicechat", target="tool", modality="tool_call"),
        GraphEdgeConfig(source="tool", target="voicechat", modality="tool_response"),
    ],
)
```

The graph is the `graph` field of `PipelineConfig`; every stage it names must
be one of the pipeline's stages, and every node must be a linear chain ending
at a terminal stage. `sglang_omni.config.schema.PipelineConfig.validate_graph`
enforces this at load time.

## Runtime

`sglang_omni.pipeline.graph.GraphSession` runs in the coordinator process on
top of the session API (`open_session`, `append_session`, `session_outputs`,
`control_session`, `close_session`), so it works with either `Coordinator` or
`Client`.

- **Open** opens one session per node in declaration order; a failed open
  closes the nodes already open.
- **Input** from the client is fanned out to the nodes listed under `inputs`.
  Each node has an inbox; one forwarder per node renumbers chunks into that
  node's own sequence, so client input and edge input interleave safely.
- **Routing**: one reader per node consumes its outputs. A chunk whose
  `(source, modality)` matches a data edge is appended to the target node; a
  control edge delivers it as a control event; data from the `output` node
  goes to the client.
- **Close** stops input, closes nodes in reverse declaration order, then stops
  the readers. Any node failure closes the whole graph and surfaces the first
  error to the output reader.
- Nodes fed only by edges (for example a tool) have no idle timeout of their
  own; the client input nodes bound the graph's idle time.

## Control events

A control edge turns a node output (for example an `interrupt` from a VAD
node) into a `control` session operation on the target node:

```python
GraphEdgeConfig(source="vad", target="llm", modality="interrupt",
                kind="control", preempt=True)
```

- The event reaches `SessionHooks.control(session_identity, event)` on every
  stage of the target node, or on `target_stages` only.
- Control operations share the per-session arrival order with appends.
- `preempt=True` sets the cancel event of the target's in-flight appends as
  soon as the control arrives, so the running unit stops early; queued input
  still runs.
- Control is currently handled by `SessionScheduler` stages. AR stages served
  by `OmniScheduler` reject it until they implement control semantics.

## Tool node

`sglang_omni.scheduling.tool_session.create_tool_scheduler(tools)` builds a
CPU node that maps tool names to Python functions. It consumes
`{"calls": [{"name", "arguments"}]}` and emits one `tool_response` chunk
`{"responses": [{"name", "response"}]}`. Unknown tools and bad arguments come
back as error responses so the model can recover.
