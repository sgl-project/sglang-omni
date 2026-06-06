from __future__ import annotations

import json
from pathlib import Path

from .models import Stage


class StageRegistry:
    def __init__(self, stages: list[Stage]) -> None:
        self._stages = tuple(sorted(stages, key=lambda stage: (stage.order, stage.stage_id)))
        self._by_id = {stage.stage_id: stage for stage in self._stages}

    @classmethod
    def load(cls, path: Path) -> "StageRegistry":
        data = json.loads(path.read_text(encoding="utf-8"))
        stages = []
        for raw in data.get("stages", []):
            stages.append(
                Stage(
                    stage_id=raw["stage_id"],
                    check_name=raw["check_name"],
                    order=int(raw["order"]),
                    depends_on=tuple(raw.get("depends_on", [])),
                    capacity_group=raw.get("capacity_group", "self-hosted-gpu"),
                    workflow_id=raw.get("workflow_id", "scheduler-gpu-stage.yaml"),
                    workflow_ref=raw.get("workflow_ref", "main"),
                    timeout_minutes=int(raw.get("timeout_minutes", 60)),
                    commands=tuple(raw.get("commands", [])),
                )
            )
        return cls(stages)

    def all(self) -> tuple[Stage, ...]:
        return self._stages

    def get(self, stage_id: str) -> Stage:
        try:
            return self._by_id[stage_id]
        except KeyError as exc:
            raise KeyError(f"Unknown stage_id: {stage_id}") from exc

    def has(self, stage_id: str) -> bool:
        return stage_id in self._by_id
