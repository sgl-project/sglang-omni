# SPDX-License-Identifier: Apache-2.0
"""Natural boundary selection for bounded Ming speech fragments."""

from __future__ import annotations

from sglang_omni.models.ming3_audio.text_processing_contract import TextToken

MIN_EARLY_FLUSH_RATIO = 3


def choose_fragment_end(
    tokens: list[TextToken],
    fragment_start: int,
    fit_end: int,
    max_semantic_length: int,
) -> int:
    early_flush_semantic_length = max(1, max_semantic_length // MIN_EARLY_FLUSH_RATIO)
    minimum_title_body_semantic_length = max(1, early_flush_semantic_length // 2)
    semantic_length = 0
    newline_boundary = 0
    paragraph_boundary = 0
    title_boundary = 0
    sentence_boundary = 0
    clause_boundary = 0
    whitespace_boundary = 0
    line_semantic_length = 0
    line_start = fragment_start
    for candidate_end in range(fragment_start + 1, fit_end + 1):
        token = tokens[candidate_end - 1]
        semantic_length += token.semantic_length
        line_semantic_length += token.semantic_length
        boundary_rank = boundary_rank_after(tokens, candidate_end - 1)
        if boundary_rank == 0:
            if candidate_end < fit_end and tokens[candidate_end].text == "\n":
                pass
            elif (
                candidate_end > fragment_start + 1
                and tokens[candidate_end - 2].text == "\n"
            ):
                if not paragraph_boundary:
                    paragraph_boundary = candidate_end
                else:
                    pass
            else:
                following_line_semantic_length = 0
                following_line_end = candidate_end
                while (
                    following_line_end < fit_end
                    and tokens[following_line_end].text != "\n"
                ):
                    following_line_semantic_length += tokens[
                        following_line_end
                    ].semantic_length
                    following_line_end += 1
                is_numbered_heading = (
                    candidate_end - line_start >= 3
                    and tokens[line_start].text.isdecimal()
                    and tokens[line_start + 1].text == "."
                    and tokens[line_start + 2].text.isspace()
                )
                if not title_boundary and (
                    is_numbered_heading
                    or (
                        line_semantic_length < early_flush_semantic_length
                        and following_line_semantic_length
                        >= minimum_title_body_semantic_length
                    )
                ):
                    title_boundary = candidate_end
                elif (
                    semantic_length >= early_flush_semantic_length
                    or following_line_semantic_length >= early_flush_semantic_length
                ):
                    newline_boundary = candidate_end
                else:
                    pass
        elif (
            boundary_rank == 1
            and semantic_length >= early_flush_semantic_length
            and not sentence_boundary
        ):
            sentence_boundary = candidate_end
        elif (
            boundary_rank == 2
            and semantic_length >= early_flush_semantic_length
            and not clause_boundary
        ):
            clause_boundary = candidate_end
        elif boundary_rank == 3:
            whitespace_boundary = candidate_end
        else:
            pass
        if boundary_rank == 0:
            line_semantic_length = 0
            line_start = candidate_end
        else:
            pass
    if paragraph_boundary and (
        not title_boundary or paragraph_boundary < title_boundary
    ):
        return paragraph_boundary
    elif title_boundary:
        return title_boundary
    elif paragraph_boundary:
        return paragraph_boundary
    elif newline_boundary:
        return newline_boundary
    elif sentence_boundary:
        return sentence_boundary
    elif clause_boundary:
        return clause_boundary
    elif whitespace_boundary:
        return whitespace_boundary
    else:
        return fit_end


def boundary_rank_after(tokens: list[TextToken], token_index: int) -> int:
    token_text = tokens[token_index].text
    if token_text in "。！？!?.":
        if (
            token_index + 1 < len(tokens)
            and tokens[token_index + 1].text in "”’」』）】》)]}\"'»"
        ):
            return 4
        else:
            return 1
    elif token_text in "”’」』）】》)]}\"'»":
        if token_index > 0 and tokens[token_index - 1].text in "。！？!?.":
            return 1
        else:
            return 4
    elif token_text == "\n":
        return 0
    elif token_text in "，,、；;：:":
        return 2
    elif token_text.isspace():
        return 3
    else:
        return 4
