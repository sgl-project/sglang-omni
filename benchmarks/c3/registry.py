# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Pinned revision constants, subtask registry, and metadata SHA-256 digests for C3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

C3_DATASET_ID = "ChengqianMa/C3"
C3_DATASET_REVISION = "451b53144ab4276a6a020609c16e324bf91e8410"

C3Language = Literal["en", "zh"]
C3LanguageSelection = Literal["all", "en", "zh"]
C3Phenomenon = Literal[
    "phonological", "semantic", "omission", "coreference", "multi-turn"
]
C3SubtaskName = Literal[
    "coreference_detection",
    "coreference_resolution",
    "multiturn",
    "omission_completion",
    "omission_detection",
    "phonological_generation",
    "phonological_generation_heteronym",
    "phonological_generation_pause",
    "phonological_generation_tone",
    "phonological_heterograph",
    "phonological_heteronym",
    "phonological_intonation",
    "phonological_pause",
    "phonological_stress",
    "phonological_tone",
    "semantic",
    "semantic_lexical",
    "semantic_syntactic",
]
C3JudgeTemplateId = Literal[
    "ambiguity_phonological",
    "ambiguity_phonological_heterograph",
    "ambiguity_phonological_intonation",
    "ambiguity_phonological_pause",
    "ambiguity_phonological_stress",
    "ambiguity_semantic",
    "ambiguity_semantic_lexical",
    "ambiguity_semantic_syntactic",
    "context_dependency_coreference_detection",
    "context_dependency_coreference_resolution",
    "context_dependency_multi_turn",
    "context_dependency_omission_completion",
    "context_dependency_omission_detection",
]
C3EvaluationScope = Literal["auto", "single_turn", "full", "subset"]
C3OutputModality = Literal["text", "audio"]
C3VerificationScope = Literal[
    "metadata_and_selected_audio_sha256",
    "metadata_and_audio_headers",
    "metadata_only",
]
C3PhonologicalScopeLabel = Literal[
    "full_with_human_generation",
    "includes_human_generation",
    "understanding_only",
]
C3MacroMetricName = Literal[
    "c3_unverified_dataset_macro_accuracy",
    "c3_paper_full_macro_accuracy",
    "c3_auto_macro_accuracy",
    "c3_subset_macro_accuracy",
]

C3_LANGUAGES: tuple[C3Language, C3Language] = ("en", "zh")
C3_PHENOMENA: tuple[C3Phenomenon, ...] = (
    "phonological",
    "semantic",
    "omission",
    "coreference",
    "multi-turn",
)

SAFE_SUBTASK_ID_RE = re.compile(r"^[a-z0-9_]+/[a-z0-9_]+/[0-9]+$")


@dataclass(frozen=True, kw_only=True)
class C3SubtaskSpec:
    language: C3Language
    phenomenon: C3Phenomenon
    subtask: C3SubtaskName
    category_path: str
    reference_json_relative_path: str
    speech_directory_relative_path: str
    judge_template_id: C3JudgeTemplateId | None
    expected_instance_count: int
    is_multiturn: bool = False
    requires_human_audio_annotation: bool = False


PHENOMENA_BY_NAME: dict[str, C3Phenomenon] = {
    "phonological": "phonological",
    "semantic": "semantic",
    "omission": "omission",
    "coreference": "coreference",
    "multi-turn": "multi-turn",
}
SUBTASK_NAMES_BY_TEXT: dict[str, C3SubtaskName] = {
    "coreference_detection": "coreference_detection",
    "coreference_resolution": "coreference_resolution",
    "multiturn": "multiturn",
    "omission_completion": "omission_completion",
    "omission_detection": "omission_detection",
    "phonological_generation": "phonological_generation",
    "phonological_generation_heteronym": "phonological_generation_heteronym",
    "phonological_generation_pause": "phonological_generation_pause",
    "phonological_generation_tone": "phonological_generation_tone",
    "phonological_heterograph": "phonological_heterograph",
    "phonological_heteronym": "phonological_heteronym",
    "phonological_intonation": "phonological_intonation",
    "phonological_pause": "phonological_pause",
    "phonological_stress": "phonological_stress",
    "phonological_tone": "phonological_tone",
    "semantic": "semantic",
    "semantic_lexical": "semantic_lexical",
    "semantic_syntactic": "semantic_syntactic",
}
JUDGE_TEMPLATE_IDS_BY_TEXT: dict[str, C3JudgeTemplateId] = {
    "ambiguity_phonological": "ambiguity_phonological",
    "ambiguity_phonological_heterograph": "ambiguity_phonological_heterograph",
    "ambiguity_phonological_intonation": "ambiguity_phonological_intonation",
    "ambiguity_phonological_pause": "ambiguity_phonological_pause",
    "ambiguity_phonological_stress": "ambiguity_phonological_stress",
    "ambiguity_semantic": "ambiguity_semantic",
    "ambiguity_semantic_lexical": "ambiguity_semantic_lexical",
    "ambiguity_semantic_syntactic": "ambiguity_semantic_syntactic",
    "context_dependency_coreference_detection": "context_dependency_coreference_detection",
    "context_dependency_coreference_resolution": "context_dependency_coreference_resolution",
    "context_dependency_multi_turn": "context_dependency_multi_turn",
    "context_dependency_omission_completion": "context_dependency_omission_completion",
    "context_dependency_omission_detection": "context_dependency_omission_detection",
}

COMPACT_SUBTASK_LINES: tuple[str, ...] = (
    "en|phonological|phonological_heterograph|/ambiguity/phonological/heterograph|english/text/ambiguity/phonological/heterograph.json|english/speech/ambiguity/phonological/heterograph|ambiguity_phonological_heterograph|5|0|0",
    "en|phonological|phonological_pause|/ambiguity/phonological/pause|english/text/ambiguity/phonological/pause.json|english/speech/ambiguity/phonological/pause|ambiguity_phonological_pause|2|0|0",
    "en|phonological|phonological_stress|/ambiguity/phonological/stress|english/text/ambiguity/phonological/stress.json|english/speech/ambiguity/phonological/stress|ambiguity_phonological_stress|6|0|0",
    "en|phonological|phonological_intonation|/ambiguity/phonological/intonation|english/text/ambiguity/phonological/intonation.json|english/speech/ambiguity/phonological/intonation|ambiguity_phonological_intonation|4|0|0",
    "en|semantic|semantic_lexical|/ambiguity/semantic/lexical|english/text/ambiguity/semantic/lexical.json|english/speech/ambiguity/semantic/lexical|ambiguity_semantic_lexical|13|0|0",
    "en|semantic|semantic_syntactic|/ambiguity/semantic/syntactic|english/text/ambiguity/semantic/syntactic.json|english/speech/ambiguity/semantic/syntactic|ambiguity_semantic_syntactic|38|0|0",
    "en|omission|omission_detection|/context-dependency/omission-detection|english/text/context-dependency/omission.json|english/speech/context-dependency/omission-detection|context_dependency_omission_detection|51|0|0",
    "en|omission|omission_completion|/context-dependency/omission-completion|english/text/context-dependency/omission.json|english/speech/context-dependency/omission-completion|context_dependency_omission_completion|51|0|0",
    "en|coreference|coreference_detection|/context-dependency/coreference-detection|english/text/context-dependency/coreference.json|english/speech/context-dependency/coreference-detection|context_dependency_coreference_detection|270|0|0",
    "en|coreference|coreference_resolution|/context-dependency/coreference-resolution|english/text/context-dependency/coreference.json|english/speech/context-dependency/coreference-resolution|context_dependency_coreference_resolution|270|0|0",
    "en|multi-turn|multiturn|/context-dependency/multi-turn|english/text/context-dependency/multi-turn.json|english/speech/context-dependency/multi-turn|context_dependency_multi_turn|34|1|0",
    "zh|phonological|phonological_pause|/ambiguity/phonological/pause|chinese/text/ambiguity/phonological/pause.json|chinese/speech/ambiguity/phonological/pause|ambiguity_phonological|5|0|0",
    "zh|phonological|phonological_heteronym|/ambiguity/phonological/heteronym|chinese/text/ambiguity/phonological/heteronym.json|chinese/speech/ambiguity/phonological/heteronym|ambiguity_phonological|2|0|0",
    "zh|phonological|phonological_heterograph|/ambiguity/phonological/heterograph|chinese/text/ambiguity/phonological/heterograph.json|chinese/speech/ambiguity/phonological/heterograph|ambiguity_phonological|3|0|0",
    "zh|phonological|phonological_tone|/ambiguity/phonological/tone|chinese/text/ambiguity/phonological/tone.json|chinese/speech/ambiguity/phonological/tone|ambiguity_phonological|5|0|0",
    "zh|semantic|semantic|/ambiguity/semantic|chinese/text/ambiguity/semantic.json|chinese/speech/ambiguity/semantic|ambiguity_semantic|118|0|0",
    "zh|omission|omission_detection|/context-dependency/omission-detection|chinese/text/context-dependency/omission.json|chinese/speech/context-dependency/omission-detection|context_dependency_omission_detection|35|0|0",
    "zh|omission|omission_completion|/context-dependency/omission-completion|chinese/text/context-dependency/omission.json|chinese/speech/context-dependency/omission-completion|context_dependency_omission_completion|35|0|0",
    "zh|coreference|coreference_detection|/context-dependency/coreference-detection|chinese/text/context-dependency/coreference.json|chinese/speech/context-dependency/coreference-detection|context_dependency_coreference_detection|30|0|0",
    "zh|coreference|coreference_resolution|/context-dependency/coreference-resolution|chinese/text/context-dependency/coreference.json|chinese/speech/context-dependency/coreference-resolution|context_dependency_coreference_resolution|30|0|0",
    "zh|multi-turn|multiturn|/context-dependency/multi-turn|chinese/text/context-dependency/multi-turn.json|chinese/speech/context-dependency/multi-turn|context_dependency_multi_turn|38|1|0",
    "en|phonological|phonological_generation|/ambiguity/phonological/generation|english/text/ambiguity/phonological/generation.json|english/speech/ambiguity/phonological/generation||12|0|1",
    "zh|phonological|phonological_generation_pause|/ambiguity/phonological/generation/pause|chinese/text/ambiguity/phonological/generation/pause.json|chinese/speech/ambiguity/phonological/generation/pause||10|0|1",
    "zh|phonological|phonological_generation_heteronym|/ambiguity/phonological/generation/heteronym|chinese/text/ambiguity/phonological/generation/heteronym.json|chinese/speech/ambiguity/phonological/generation/heteronym||4|0|1",
    "zh|phonological|phonological_generation_tone|/ambiguity/phonological/generation/tone|chinese/text/ambiguity/phonological/generation/tone.json|chinese/speech/ambiguity/phonological/generation/tone||8|0|1",
)

C3_SUBTASK_SPECS: tuple[C3SubtaskSpec, ...] = tuple(
    C3SubtaskSpec(
        language=("en" if subtask_columns[0] == "en" else "zh"),
        phenomenon=PHENOMENA_BY_NAME[subtask_columns[1]],
        subtask=SUBTASK_NAMES_BY_TEXT[subtask_columns[2]],
        category_path=subtask_columns[3],
        reference_json_relative_path=subtask_columns[4],
        speech_directory_relative_path=subtask_columns[5],
        judge_template_id=(
            JUDGE_TEMPLATE_IDS_BY_TEXT[subtask_columns[6]]
            if subtask_columns[6]
            else None
        ),
        expected_instance_count=int(subtask_columns[7]),
        is_multiturn=(subtask_columns[8] == "1"),
        requires_human_audio_annotation=(subtask_columns[9] == "1"),
    )
    for raw_line in COMPACT_SUBTASK_LINES
    for subtask_columns in (raw_line.split("|"),)
)


def validate_c3_instance_id(instance_id: str) -> str:
    if (
        not isinstance(instance_id, str)
        or SAFE_SUBTASK_ID_RE.fullmatch(instance_id) is None
    ):
        raise ValueError(f"Unsafe C3 instance_id: {instance_id!r}")
    else:
        return instance_id


def select_c3_subtask_specs(
    *,
    language: C3LanguageSelection = "all",
    include_human_generation: bool = False,
    include_multiturn: bool = True,
    subtask_filter: tuple[str, ...] | None = None,
) -> list[C3SubtaskSpec]:
    """Select C3 subtasks while validating every supplied filter token."""
    allowed_languages = C3_LANGUAGES if language == "all" else (language,)
    all_qualified_names = {
        f"{subtask_spec.language}/{subtask_spec.subtask}": subtask_spec
        for subtask_spec in C3_SUBTASK_SPECS
    }
    all_unqualified_names: dict[str, list[C3SubtaskSpec]] = {}
    for subtask_spec in C3_SUBTASK_SPECS:
        all_unqualified_names.setdefault(subtask_spec.subtask, []).append(subtask_spec)

    if subtask_filter is not None:
        if not subtask_filter:
            raise ValueError("--subtasks must not be empty when specified")
        else:
            pass
        for filter_token in subtask_filter:
            if "/" in filter_token:
                matched_spec = all_qualified_names.get(filter_token)
                if matched_spec is None:
                    raise ValueError(
                        f"Unknown C3 subtask {filter_token!r}; valid qualified names are {sorted(all_qualified_names.keys())}"
                    )
                elif matched_spec.language not in allowed_languages:
                    raise ValueError(
                        f"Subtask {filter_token!r} conflicts with --language {language!r}"
                    )
                elif matched_spec.is_multiturn and not include_multiturn:
                    raise ValueError(
                        f"Multi-turn subtask {filter_token!r} conflicts with single-turn scope"
                    )
                elif (
                    matched_spec.requires_human_audio_annotation
                    and not include_human_generation
                ):
                    raise ValueError(
                        f"Human-scored generation subtask {filter_token!r} is excluded under automated scope; "
                        "pass --scope full or --scope subset (with --modalities text,audio)"
                    )
                else:
                    pass
            else:
                candidate_specs = all_unqualified_names.get(filter_token)
                if candidate_specs is None:
                    raise ValueError(
                        f"Unknown C3 subtask {filter_token!r}; valid subtask names are {sorted(all_unqualified_names.keys())}"
                    )
                elif not any(
                    subtask_spec.language in allowed_languages
                    and (include_multiturn or not subtask_spec.is_multiturn)
                    and (
                        include_human_generation
                        or not subtask_spec.requires_human_audio_annotation
                    )
                    for subtask_spec in candidate_specs
                ):
                    raise ValueError(
                        f"Subtask {filter_token!r} has no eligible splits under language={language!r}"
                    )
                else:
                    pass
    else:
        pass

    filter_token_set = set(subtask_filter) if subtask_filter is not None else None
    selected_subtask_specs: list[C3SubtaskSpec] = [
        subtask_spec
        for subtask_spec in C3_SUBTASK_SPECS
        if subtask_spec.language in allowed_languages
        and (
            include_human_generation or not subtask_spec.requires_human_audio_annotation
        )
        and (include_multiturn or not subtask_spec.is_multiturn)
        and (
            filter_token_set is None
            or f"{subtask_spec.language}/{subtask_spec.subtask}" in filter_token_set
            or subtask_spec.subtask in filter_token_set
        )
    ]
    if not selected_subtask_specs:
        raise ValueError("No C3 subtasks matched the requested selection")
    else:
        return selected_subtask_specs
