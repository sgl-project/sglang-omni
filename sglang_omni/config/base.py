# SPDX-License-Identifier: Apache-2.0
"""Recursive conflict checks for resolved configuration models."""

from typing import final

from pydantic import BaseModel


class ConfigModel(BaseModel):
    """Define local rules in check_local_conflicts; nested models are visited automatically."""

    @final
    def check_conflicts(self) -> None:
        """Check nested models before their owners, without changing configuration values."""

        def visit(model: BaseModel) -> None:
            pending = [getattr(model, name) for name in type(model).model_fields]
            while pending:
                value = pending.pop()
                if isinstance(value, BaseModel):
                    visit(value)
                elif isinstance(value, dict):
                    pending.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    pending.extend(value)
                else:
                    pass
            if isinstance(model, ConfigModel):
                model.check_local_conflicts()
            else:
                pass

        visit(self)

    def check_local_conflicts(self) -> None:
        """Override to check this model's relationships after its children pass."""
