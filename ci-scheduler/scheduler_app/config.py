from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the scheduler service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_id: str | None = Field(default=None, alias="GITHUB_APP_ID")
    private_key: str | None = Field(default=None, alias="GITHUB_PRIVATE_KEY")
    private_key_path: Path | None = Field(default=None, alias="GITHUB_PRIVATE_KEY_PATH")
    fallback_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    webhook_secret: str | None = Field(default=None, alias="GITHUB_WEBHOOK_SECRET")

    default_repo: str = Field(default="sgl-project/sglang-omni", alias="GITHUB_REPOSITORY")
    dispatch_workflow_id: str = Field(
        default="scheduler-gpu-stage.yaml",
        alias="DISPATCH_WORKFLOW_ID",
    )
    dispatch_ref: str = Field(default="main", alias="DISPATCH_REF")

    database_path: Path = Field(default=Path("/data/ci-scheduler.sqlite3"), alias="DATABASE_PATH")
    stages_path: Path = Field(default=Path("stages.json"), alias="STAGES_PATH")

    runner_capacity: int = Field(default=1, alias="RUNNER_CAPACITY")
    run_label: str = Field(default="run-ci", alias="RUN_LABEL")
    high_priority_label: str = Field(default="high-priority", alias="HIGH_PRIORITY_LABEL")

    callback_token: str | None = Field(default=None, alias="SCHEDULER_CALLBACK_TOKEN")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")

    @property
    def github_private_key(self) -> str | None:
        if self.private_key:
            return self.private_key.replace("\\n", "\n")
        if self.private_key_path:
            return self.private_key_path.read_text(encoding="utf-8")
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
