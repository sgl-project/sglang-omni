# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Official English and Chinese LLM-judge prompt templates and verdict parsing for C3."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from benchmarks.c3.containment import read_bytes_nofollow
from benchmarks.c3.registry import (
    C3JudgeTemplateId,
    C3Language,
    validate_c3_instance_id,
)
from benchmarks.c3.schemas import (
    C3ChineseAmbiguityReference,
    C3ChineseContextReference,
    C3ChineseMultiturnReference,
    C3EnglishCoreferenceReference,
    C3EnglishExplanationReference,
    C3EnglishHeterographReference,
    C3EnglishMultiturnReference,
    C3EnglishSyntacticReference,
)
from benchmarks.dataset.c3 import C3InstanceRecord

C3_SYSTEM_PROMPTS: dict[C3Language, str] = {
    "en": (
        "You are an expert proficient in analyzing linguistic and phonetic phenomena, "
        "with extensive research experience in various aspects of language and speech, "
        "including semantic comprehension, homophone differentiation, polysemy disambiguation, "
        "analysis of special sentence structures, ambiguity identification, grammatical "
        "structure analysis and so on in English spoken conversations."
    ),
    "zh": (
        "你是一个擅长分析语言学、语音学现象的专家，在中文语音对话中的语义理解、同音词辨别、"
        "一词多义辨析、特殊句式分析、歧义辨别、语法结构分析等语言和语音中的各个领域都拥有丰富的研究经验。"
    ),
}

EN_VERDICT_FOOTER = (
    "If the student's {subject} is correct, output 'The answer is: yes.' at the end; "
    "otherwise, output 'The answer is: no.' at the end."
)
EN_EXPLANATION_RUBRIC = (
    "The correct understanding of this sentence is:\n```\n{notation}\n```\n"
    "A student has provided their own interpretation of this sentence:\n```\n{answer}\n```\n"
    "Please analyze whether the student's interpretation is correct. First, understand "
    "the ambiguous sentence and its corresponding correct interpretation, and then "
    "analyze the student's interpretation.\n"
    + EN_VERDICT_FOOTER.format(subject="interpretation")
)
EN_QA_RUBRIC = (
    "Regarding this sentence, there is a question:\n```\n{problem}\n```\n"
    "The standard answer to this question is:\n```\n{standard_answer}\n```\n"
    "A student provided their own answer to the question:\n```\n{answer}\n```\n"
    "Please analyze whether the student's answer is correct. First, understand the "
    "question and the standard answer, and then analyze the student's answer.\n"
    + EN_VERDICT_FOOTER.format(subject="answer")
)
ZH_TERMINAL_FORMAT_FOOTER = "请在最后一行回答是或否。回答时请遵循格式：\n最终答案为：是。\n或者\n最终答案为：否。"
ZH_AMBIGUITY_TEMPLATE = (
    "这里有一段考察语义理解的材料，材料内容如下：\n```\n{content}\n```\n"
    "请问这段材料应该如何理解？\n正确的答案为：\n```\n{annotation}\n```\n"
    "有一名学生针对如何理解该材料这一问题，给出了如下答案：\n```\n{answer}\n```\n"
    "请问该学生的理解是否正确？在分析过程中，请先理解题目和正确答案，随后分析学生的答案是否正确。\n"
    + ZH_TERMINAL_FORMAT_FOOTER
)

C3_EVALUATION_TEMPLATES: dict[C3Language, dict[C3JudgeTemplateId, str]] = {
    "en": {
        "ambiguity_phonological_heterograph": (
            "This is a sentence that contains ambiguity caused by homophony when read:\n```\n{content}\n```\n"
            + EN_QA_RUBRIC.replace("{standard_answer}", "{notation}")
        ),
        "ambiguity_phonological_pause": (
            "This is a sentence that contains ambiguity caused by pauses when read:\n```\n{content}\n```\n"
            + EN_EXPLANATION_RUBRIC
        ),
        "ambiguity_phonological_stress": (
            "This is a sentence that contains ambiguity caused by stress when read:\n```\n{content}\n```\n"
            + EN_EXPLANATION_RUBRIC
        ),
        "ambiguity_phonological_intonation": (
            "This is a sentence that contains ambiguity caused by intonation when read:\n```\n{content}\n```\n"
            + EN_EXPLANATION_RUBRIC
        ),
        "ambiguity_semantic_lexical": (
            "This is a sentence that contains ambiguity caused by polysemy:\n```\n{content}\n```\n"
            + EN_EXPLANATION_RUBRIC
        ),
        "ambiguity_semantic_syntactic": (
            "This is a sentence that contains ambiguity caused by syntactic structure:\n```\n{content}\n```\n"
            "The correct understanding of this sentence is:\n```\n{ambiguity}\n```\n{interpretations}\n"
            "A student has provided their own interpretation of this sentence:\n```\n{answer}\n```\n"
            "Please analyze whether the student's interpretation is correct. First, understand "
            "the ambiguous sentence and its corresponding correct interpretation, and then "
            "analyze the student's interpretation.\n"
            + EN_VERDICT_FOOTER.format(subject="interpretation")
        ),
        "context_dependency_omission_detection": (
            "This is a sentence that contains ambiguity caused by omission:\n```\n{content}\n```\n"
            "When asked whether there was any omission in the sentence, one student gave his/her answer:\n```\n{answer}\n```\n"
            "Check if the student correctly identified the omission in the sentence.\n"
            + EN_VERDICT_FOOTER.format(subject="answer")
        ),
        "context_dependency_omission_completion": (
            "This is a sentence that contains ambiguity caused by omission:\n```\n{content}\n```\n"
            "How should we resolve the ambiguity of this sentence and complete the missing content? "
            "This is the standard answer:\n```\n{notation}\n```\n"
            "A student has provided their own answer regarding this problem:\n```\n{answer}\n```\n"
            "Compared to the standard answer, did the student restore all the omitted content? "
            "First, understand the ambiguous sentence, the question, and the correct answer, "
            "and then analyze the student's answer. If the student's answer includes all the "
            "restored content that is present in the standard answer, then the student's answer "
            "is considered correct.\n" + EN_VERDICT_FOOTER.format(subject="answer")
        ),
        "context_dependency_coreference_detection": (
            "This is a sentence that contains ambiguity caused by coreference:\n```\n{content}\n```\n"
            "When asked whether there was any coreference in the sentence, one student gave his/her answer:\n```\n{answer}\n```\n"
            "Assess whether the student has successfully detected the presence of a coreference phenomenon in the sentence.\n"
            "If the student has identified the coreference phenomenon, their response is deemed correct, "
            "irrespective of the accuracy of the content they identified as being referred to.\n"
            "Conversely, if the student fails to identify the referential phenomenon, their response is considered incorrect, "
            "regardless of the reasons provided.\n"
            + EN_VERDICT_FOOTER.format(subject="answer")
        ),
        "context_dependency_coreference_resolution": (
            "This is a sentence that contains ambiguity caused by coreference:\n```\n{content}\n```\n"
            + EN_QA_RUBRIC.replace("{standard_answer}", "{correct_answer}")
        ),
        "context_dependency_multi_turn": (
            "This is a conversation between A and B. Please understand this dialogue, and I will pose a question after the dialogue:\n{dialogues}\n"
            "Please analyze whether B's response in the last round of the dialogue is correct. "
            "It is known that the question posed by A in the last round of the dialogue is the same as the question posed by A in the first round. "
            "If B's response in the last round is consistent with B's response in the first round and does not conflict with the content of the dialogue in between, "
            "then B's response in the last round is considered correct. Please first understand B's responses in both the first and last rounds of the dialogue, "
            "compare whether the two responses are consistent, and finally analyze whether B's response in the last round does not conflict with the content of the dialogue in between.\n"
            "If B's response in the last round is correct, answer 'The answer is: yes.' at the end; otherwise, answer 'The answer is: no.' at the end."
        ),
    },
    "zh": {
        "ambiguity_phonological": ZH_AMBIGUITY_TEMPLATE,
        "ambiguity_semantic": ZH_AMBIGUITY_TEMPLATE,
        "context_dependency_omission_detection": (
            "这里有一段对话，对话中最后一句存在省略现象，对话内容如下：\n```\n{dialogue_content}\n```\n"
            "对于这句话中是否包含省略现象，一名学生给出了自己的回答如下：\n```\n{answer}\n```\n"
            "请分析，该学生是否正确识别出了对话中的省略现象？当该学生识别出最后一句话中存在的省略现象时，我们认为是；"
            "当该学生没能识别出最后一句话中存在的省略现象时，我们认为否。\n"
            + ZH_TERMINAL_FORMAT_FOOTER
        ),
        "context_dependency_omission_completion": (
            "这里有一段对话，对话中最后一句存在省略现象，对话内容如下：\n```\n{dialogue_content}\n```\n"
            "请补全被省略的内容，给出省略内容被补全后的完整语句。\n将其中省略内容全部补全后的正确答案为：\n```\n{omission_complete}\n```\n"
            "有一名学生给出这样的回答：\n```\n{answer}\n```\n"
            "请分析：该学生的回答是否正确？正确的标准是：学生的回答将正确答案中被补全的所有省略内容，都进行了补全。\n"
            + ZH_TERMINAL_FORMAT_FOOTER
        ),
        "context_dependency_coreference_detection": (
            "这里有一段对话，对话中最后一句存在使用代词指代具体内容的现象，对话内容如下：\n```\n{dialogue_content}\n```\n"
            "对于这句话中是否包含指代现象，一名学生给出了自己的回答如下：\n```\n{answer}\n```\n"
            "请分析，该学生是否正确识别出了对话中的指代现象？当该学生识别出最后一句话中存在的指代现象时，我们认为是；"
            "当该学生没能识别出最后一句话中存在的指代现象时，我们认为否。\n"
            + ZH_TERMINAL_FORMAT_FOOTER
        ),
        "context_dependency_coreference_resolution": (
            "这里有一段对话，对话中最后一句存在使用代词指代具体内容的现象，对话内容如下：\n```\n{dialogue_content}\n```\n"
            "请把代词替换成代词指代的内容，给出所有代词被替换后的完整语句。\n将其中代词全部替换后的正确答案为：\n```\n{coreference_complete}\n```\n"
            "有一名学生给出这样的回答：\n```\n{answer}\n```\n"
            "请分析：该学生的回答是否正确？正确的标准是：学生的回答将正确答案中被替换的所有代词，都进行了替换。\n"
            + ZH_TERMINAL_FORMAT_FOOTER
        ),
        "context_dependency_multi_turn": (
            "这里有甲和乙的一段对话，请听完这段对话，并回答我在对话之后提出的问题：\n{dialogues}\n"
            "对话结束，请分析乙在最后一轮对话中的回答是否正确。已知最后一轮对话中甲提出的问题，和第一轮对话中甲提出的问题是一样的。"
            "如果乙最后一轮对话中给出的回答，和乙第一轮对话中给出的回答一致，且和两人中间对话的内容不冲突，则说明乙在最后一轮对话中的回答是正确的。"
            "请先理解乙在第一轮和最后一轮对话中的回答，再对比两次回答是否一致，接着分析乙最后一轮的回答是否不与中间对话的内容冲突。"
            "最后分析乙最后一轮对话中的回答是否正确。\n" + ZH_TERMINAL_FORMAT_FOOTER
        ),
    },
}

ENGLISH_VERDICT_RE = re.compile(
    r"(?:^|\n|\s|[.!?;:,，。！？；：])the answer is:\s*(yes|no)[.!。]?\s*\Z",
    re.IGNORECASE,
)
CHINESE_VERDICT_RE = re.compile(
    r"(?:^|\n|\s|[，。！？；：,.!?;:])最终答案为[：:]\s*(是|否)[。.!]?\s*\Z"
)


TrimmedNonBlankIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class C3ManualAnnotationInputRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    correct: bool | str
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotator_id: TrimmedNonBlankIdentifier | None = None


class C3ManualGenerationAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    correct: bool
    prediction_sha256: str
    generated_audio_sha256: str
    annotator_id: str | None = None


def compute_c3_prompt_bundle_sha256() -> str:
    canonical_payload = json.dumps(
        {
            "system_prompts": C3_SYSTEM_PROMPTS,
            "evaluation_prompts": C3_EVALUATION_TEMPLATES,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def parse_c3_judge_verdict(raw_output: str, language: C3Language) -> bool | None:
    """Return True or False for a valid terminal verdict, or None when missing or malformed."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None
    else:
        verdict_pattern = ENGLISH_VERDICT_RE if language == "en" else CHINESE_VERDICT_RE
        verdict_match = verdict_pattern.search(raw_output)
        if verdict_match is None:
            return None
        else:
            return verdict_match.group(1).lower() in ("yes", "是")


def build_c3_judge_messages(
    instance: C3InstanceRecord,
    assistant_turn_responses: tuple[str, ...] | list[str],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) matching official C3 evaluation formatting byte-for-byte."""
    if instance.requires_human_audio_annotation or instance.judge_template_id is None:
        raise ValueError(
            f"Instance {instance.instance_id} requires human audio annotation and has no LLM judge template"
        )
    elif len(assistant_turn_responses) != len(instance.user_turn_texts):
        raise ValueError(
            f"Instance {instance.instance_id} expected {len(instance.user_turn_texts)} "
            f"assistant turn responses, got {len(assistant_turn_responses)}"
        )
    else:
        pass

    system_prompt = C3_SYSTEM_PROMPTS[instance.language]
    template = C3_EVALUATION_TEMPLATES[instance.language][instance.judge_template_id]
    reference = instance.reference_record

    if isinstance(reference, C3EnglishMultiturnReference):
        formatted_turns = [
            f"\nA:\n```\n{user_text}\n```\nB:\n```\n{assistant_text}\n```"
            for user_text, assistant_text in zip(
                instance.user_turn_texts, assistant_turn_responses
            )
        ]
        return system_prompt, template.format(dialogues="".join(formatted_turns))
    elif isinstance(reference, C3ChineseMultiturnReference):
        formatted_turns = [
            f"\n甲说：\n```\n{user_text}\n```\n乙说：\n```\n{assistant_text}\n```"
            for user_text, assistant_text in zip(
                instance.user_turn_texts, assistant_turn_responses
            )
        ]
        return system_prompt, template.format(dialogues="".join(formatted_turns))
    else:
        pass

    final_answer = assistant_turn_responses[0]
    if isinstance(reference, C3EnglishHeterographReference):
        return system_prompt, template.format(
            content=reference.content,
            problem=reference.problem,
            notation=reference.notation,
            answer=final_answer,
        )
    elif isinstance(reference, C3EnglishSyntacticReference):
        formatted_interpretations = "".join(
            f"\nThe {meaning_index + 1} meaning of this sentence is:\n```\n{interpretation_text}\n```"
            for meaning_index, interpretation_text in enumerate(
                reference.notation.interpretations
            )
        )
        return system_prompt, template.format(
            content=reference.content,
            ambiguity=reference.notation.ambiguity,
            interpretations=formatted_interpretations,
            answer=final_answer,
        )
    elif isinstance(reference, C3EnglishCoreferenceReference):
        if instance.subtask == "coreference_detection":
            return system_prompt, template.format(
                content=reference.content, answer=final_answer
            )
        else:
            return system_prompt, template.format(
                content=reference.content,
                problem=reference.problem,
                correct_answer=reference.answer,
                answer=final_answer,
            )
    elif isinstance(reference, C3EnglishExplanationReference):
        if instance.subtask == "omission_detection":
            return system_prompt, template.format(
                content=reference.content, answer=final_answer
            )
        else:
            return system_prompt, template.format(
                content=reference.content,
                notation=reference.notation,
                answer=final_answer,
            )
    elif isinstance(reference, C3ChineseContextReference):
        dialogue_content = "".join(
            f"{history_line}\n" for history_line in reference.history
        )
        if instance.subtask == "omission_completion":
            assert reference.omission_complete is not None
            return system_prompt, template.format(
                dialogue_content=dialogue_content,
                omission_complete=reference.omission_complete,
                answer=final_answer,
            )
        elif instance.subtask == "coreference_resolution":
            assert reference.coreference_complete is not None
            return system_prompt, template.format(
                dialogue_content=dialogue_content,
                coreference_complete=reference.coreference_complete,
                answer=final_answer,
            )
        else:
            return system_prompt, template.format(
                dialogue_content=dialogue_content,
                answer=final_answer,
            )
    else:
        assert isinstance(reference, C3ChineseAmbiguityReference)
        return system_prompt, template.format(
            content=reference.content,
            annotation=reference.annotation,
            answer=final_answer,
        )


def compute_c3_judge_prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    serialized_prompt_bytes = json.dumps(
        {"system": system_prompt, "user": user_prompt},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized_prompt_bytes).hexdigest()


def load_manual_generation_annotations(
    annotations_path: Path,
    *,
    allowed_human_instances: dict[str, C3InstanceRecord],
    expected_hashes_by_instance: dict[str, tuple[str, str | None]],
) -> tuple[dict[str, C3ManualGenerationAnnotation], str]:
    """Read once via O_NOFOLLOW, validate Pydantic rows against audio/prediction SHA-256, and return (annotations, file_sha256)."""
    if not allowed_human_instances:
        raise ValueError(
            "Manual generation annotations were supplied, but no human-scored phonological generation instances are selected"
        )
    else:
        pass
    expanded_path = annotations_path.expanduser()
    annotation_bytes = read_bytes_nofollow(expanded_path.parent, expanded_path.name)
    annotations_file_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    raw_payload = json.loads(annotation_bytes.decode("utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError(
            f"{annotations_path} must be a JSON object keyed by human-generation instance_id"
        )
    else:
        pass

    validated_annotations: dict[str, C3ManualGenerationAnnotation] = {}
    for raw_instance_id, raw_row in raw_payload.items():
        instance_id = validate_c3_instance_id(raw_instance_id)
        if instance_id not in allowed_human_instances:
            raise ValueError(
                f"Manual annotation key {instance_id!r} in {annotations_path} is not a selected human-scored generation instance"
            )
        elif instance_id not in expected_hashes_by_instance:
            raise ValueError(
                f"Manual annotation for {instance_id!r} has no completed prediction in predictions.json"
            )
        else:
            pass

        parsed_row = C3ManualAnnotationInputRow.model_validate(raw_row)
        expected_prediction_sha256, expected_audio_sha256 = expected_hashes_by_instance[
            instance_id
        ]
        if expected_audio_sha256 is None:
            raise ValueError(
                f"Prediction for {instance_id!r} has no verified exported WAV audio (generated_audio_sha256)"
            )
        elif parsed_row.prediction_sha256 != expected_prediction_sha256:
            raise ValueError(
                f"Manual annotation prediction_sha256 mismatch for {instance_id!r} in {annotations_path}"
            )
        elif parsed_row.generated_audio_sha256 != expected_audio_sha256:
            raise ValueError(
                f"Manual annotation generated_audio_sha256 mismatch for {instance_id!r} in {annotations_path}"
            )
        else:
            pass

        instance = allowed_human_instances[instance_id]
        if isinstance(parsed_row.correct, bool):
            parsed_verdict = parsed_row.correct
        else:
            maybe_verdict = parse_c3_judge_verdict(
                parsed_row.correct, instance.language
            )
            if maybe_verdict is None:
                raise ValueError(
                    f"Invalid verdict string for {instance_id!r} in {annotations_path}"
                )
            else:
                parsed_verdict = maybe_verdict

        validated_annotations[instance_id] = C3ManualGenerationAnnotation(
            instance_id=instance_id,
            correct=parsed_verdict,
            prediction_sha256=parsed_row.prediction_sha256,
            generated_audio_sha256=parsed_row.generated_audio_sha256,
            annotator_id=parsed_row.annotator_id,
        )
    return validated_annotations, annotations_file_sha256
