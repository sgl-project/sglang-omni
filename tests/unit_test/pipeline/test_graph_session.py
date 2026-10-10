# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from sglang_omni.config.schema import (
    GraphConfig,
    GraphEdgeConfig,
    GraphNodeConfig,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.pipeline.graph import GraphSession
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import OutputChunk
from tests.unit_test.fixtures.graph_pipeline import GRAPH, GRAPH_STAGES, audio
from tests.unit_test.fixtures.session_pipeline import event_log, pipeline, wait_until

OUTPUT_TIMEOUT_S = 5


async def next_output(outputs: AsyncIterator[OutputChunk]) -> OutputChunk:
    return await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)


@pytest.mark.asyncio
async def test_fan_out_feedback_edge_and_reverse_close(tmp_path):
    async with pipeline(tmp_path, stage_configs=GRAPH_STAGES) as (
        coordinator,
        events,
        _,
    ):
        graph = await GraphSession.open(
            coordinator, GRAPH, OmniRequest(None), session_id="g"
        )
        outputs = graph.outputs()
        await graph.append(audio(0, b"hello"))
        assert (await next_output(outputs)).payload == {"heard": "hello"}
        await graph.append(audio(1, b"call"))
        tool_call = await next_output(outputs)
        assert tool_call.modality == "tool_call"
        tool_text = await next_output(outputs)
        assert tool_text.payload == {
            "responses": [{"name": "add", "response": {"sum": 5}}]
        }
        await graph.close()
        log = event_log(events)
        assert [event[3] for event in log if event[:2] == ("append", "vad")] == [0, 1]
        assert [event[3] for event in log if event[:2] == ("append", "llm")] == [
            0,
            1,
            2,
        ]
        closes = [event[1] for event in log if event[0] == "close"]
        assert closes == ["llm", "vad"]
        assert not coordinator.sessions
        with pytest.raises(RuntimeError, match="closing"):
            await graph.append(audio(2, b"late"))


@pytest.mark.asyncio
async def test_interrupt_preempts_in_flight_unit(tmp_path):
    async with pipeline(tmp_path, stage_configs=GRAPH_STAGES) as (
        coordinator,
        events,
        _,
    ):
        graph = await GraphSession.open(
            coordinator, GRAPH, OmniRequest(None), session_id="g"
        )
        outputs = graph.outputs()
        await graph.append(audio(0, b"long"))
        await graph.append(audio(1, b"speech"))
        assert (await next_output(outputs)).payload == {"heard": "speech"}
        log = event_log(events)
        assert ("cancelled", "llm", "g:llm") in log
        assert ("control", "llm", "g:llm") in log
        assert not any(event[:2] == ("control", "vad") for event in log)
        await graph.close()


@pytest.mark.asyncio
async def test_node_failure_closes_whole_graph(tmp_path):
    async with pipeline(tmp_path, stage_configs=GRAPH_STAGES) as (
        coordinator,
        events,
        _,
    ):
        graph = await GraphSession.open(
            coordinator, GRAPH, OmniRequest(None), session_id="g"
        )
        outputs = graph.outputs()
        await graph.append(audio(0, b"fail"))
        with pytest.raises(RuntimeError, match="llm unit failed"):
            await next_output(outputs)
        await graph.close()
        await wait_until(lambda: not coordinator.sessions)
        closed = {event[1] for event in event_log(events) if event[0] == "close"}
        assert "vad" in closed


def mock_stage(name: str, next_stage: str | None = None) -> StageConfig:
    return StageConfig(
        name=name,
        process=name,
        next=next_stage,
        terminal=next_stage is None,
        factory_path="unused",
    )


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (
            GraphConfig(
                nodes={
                    "a": GraphNodeConfig(stages=["x"]),
                    "b": GraphNodeConfig(stages=["x"]),
                },
                inputs={"audio": ["a"]},
                output="a",
            ),
            "belongs to graph nodes",
        ),
        (
            GraphConfig(
                nodes={"a": GraphNodeConfig(stages=["y"])},
                inputs={"audio": ["a"]},
                output="a",
            ),
            "linear chain",
        ),
        (
            GraphConfig(
                nodes={"a": GraphNodeConfig(stages=["x"])}, inputs={}, output="a"
            ),
            "at least one client input",
        ),
        (
            GraphConfig(
                nodes={"a": GraphNodeConfig(stages=["x"])},
                inputs={"audio": ["a"]},
                output="a",
                edges=[GraphEdgeConfig(source="a", target="b", modality="text")],
            ),
            "unknown node",
        ),
        (
            GraphConfig(
                nodes={"a": GraphNodeConfig(stages=["x"])},
                inputs={"audio": ["a"]},
                output="a",
                edges=[
                    GraphEdgeConfig(
                        source="a", target="a", modality="text", preempt=True
                    )
                ],
            ),
            "only to control edges",
        ),
    ],
)
def test_graph_config_rejects_invalid_topology(graph, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig(
            model_path="mock",
            stages=[mock_stage("x"), mock_stage("y", "x")],
            graph=graph,
        )
