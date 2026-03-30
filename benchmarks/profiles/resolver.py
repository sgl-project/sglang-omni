from __future__ import annotations

from benchmarks.core.types import BenchmarkRunConfig, ResolvedProfileSelection

from .registry import BenchmarkProfileRegistry


def resolve_profile_selection(
    *,
    run_config: BenchmarkRunConfig,
    registry: BenchmarkProfileRegistry,
) -> ResolvedProfileSelection:
    profile = registry.get_by_name_or_alias(run_config.model_profile)

    case_spec = profile.get_case(run_config.case_id)
    return ResolvedProfileSelection(
        model_profile_name=profile.profile_name,
        case_spec=case_spec,
        request_family=profile.request_family,
        served_model_name=run_config.model,
        dataset_loader_name=profile.dataset_loader.name,
        model_adapter_name=profile.model_adapter.name,
        model_profile=profile,
    )
