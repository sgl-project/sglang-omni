# SPDX-License-Identifier: Apache-2.0
"""Load native tool definitions, accepted answers and WAV recordings for BFCL v3 audio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypedDict

from datasets import Audio, load_dataset

from benchmarks.metrics.bfcl_audio import ReferenceCall

DATASET_ID = "ServiceNow-AI/BFCL_v3_audio"
DATASET_REVISION = "bce0c5dd23971bacd49112427ae4ab90d0a02ab0"
SUBSETS = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")
SCHEMA_TYPES = {"dict": "object", "float": "number", "tuple": "array"}


class ParameterSchema(TypedDict, total=False):
    type: str
    properties: dict[str, ParameterSchema]
    items: ParameterSchema
    required: list[str]


class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: ParameterSchema


@dataclass(kw_only=True)
class AudioSample:
    sample_id: str
    audio_bytes: bytes
    functions: list[FunctionDefinition]
    references: list[ReferenceCall]


def normalize_schema(schema: ParameterSchema) -> None:
    schema_type = schema.get("type")
    if schema_type == "any":
        schema.pop("type")
    elif schema_type in SCHEMA_TYPES:
        schema["type"] = SCHEMA_TYPES[schema_type]
    else:
        pass
    for child in schema.get("properties", {}).values():
        normalize_schema(child)
    if "items" in schema:
        normalize_schema(schema["items"])
    else:
        pass


def load_samples(subset: str, max_samples: int | None) -> list[AudioSample]:
    dataset = load_dataset(
        DATASET_ID, f"BFCL_v3_{subset}", revision=DATASET_REVISION, split="test"
    ).cast_column("audio", Audio(decode=False))
    samples = []
    count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for row in dataset.select(range(count)):
        functions: list[FunctionDefinition] = json.loads(row["tools"])
        names = {
            function["name"]: function["name"].replace(".", "_")
            for function in functions
        }
        short_names = {
            name.rsplit(".", 1)[-1]: normalized for name, normalized in names.items()
        }
        for function in functions:
            function["name"] = names[function["name"]]
            normalize_schema(function["parameters"])
        references = [
            {
                names[name] if name in names else short_names[name]: arguments
                for name, arguments in call.items()
            }
            for call in json.loads(row["reference"])
        ]
        samples.append(
            AudioSample(
                sample_id=row["id"],
                audio_bytes=row["audio"]["bytes"],
                functions=functions,
                references=references,
            )
        )
    return samples
