import re
import string


def is_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def get_semantic_length(text):
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fa5]", text))
    english_word_count = len(re.findall(r"[a-zA-Z]+", text))
    return chinese_char_count + english_word_count


def has_valid_content(text):
    punctuation_and_whitespace = string.punctuation + string.whitespace
    for char in text:
        if char not in punctuation_and_whitespace:
            return True
    return False


def append_text_fragment(fragments, new_text, max_len, min_tail_length):
    new_text = new_text.lstrip("\u00ef\u00bc\u008c,:;" + string.whitespace)
    if not has_valid_content(new_text):
        return fragments

    if not fragments:
        fragments.append(new_text)
        return fragments

    last_fragment = fragments[-1]
    last_semantic_len = get_semantic_length(last_fragment)
    new_semantic_len = get_semantic_length(new_text)

    if last_semantic_len + new_semantic_len <= max_len:
        if (
            last_fragment.endswith(("\u3002", "\uff01", "\uff1f"))
            and new_semantic_len < min_tail_length
        ):
            fragments.append(new_text)
        else:
            separator = ""
            if not last_fragment.endswith(" ") and re.match(r"^[a-zA-Z0-9]", new_text):
                separator = " "
            fragments[-1] += separator + new_text
    else:
        fragments.append(new_text)
    return fragments


def split_long_fragment(text_fragment, max_len):
    fragment_semantic_len = get_semantic_length(text_fragment)
    if fragment_semantic_len <= max_len:
        return [text_fragment]

    fragments = []
    current_fragment = ""
    semantic_units = re.findall(
        r"([\u4e00-\u9fa5]|[a-zA-Z]+|[^a-zA-Z\u4e00-\u9fa5]+)", text_fragment
    )

    for unit in semantic_units:
        unit_semantic_len = get_semantic_length(unit)
        current_semantic_len = get_semantic_length(current_fragment)

        if current_semantic_len + unit_semantic_len <= max_len:
            current_fragment += unit
        else:
            if current_fragment:
                fragments.append(current_fragment)
            if unit_semantic_len > max_len:
                fragments.append(unit)
                current_fragment = ""
            else:
                current_fragment = unit

    if current_fragment:
        fragments.append(current_fragment)

    return fragments


def calibrate_positions(fragments, positions, original_text):
    calibrated_positions = {}
    current_global_pos = 0

    for frag_idx, fragment in enumerate(fragments):
        frag_len = len(fragment)
        found_pos = original_text.find(fragment, current_global_pos)

        if found_pos == -1:
            simplified_frag = fragment.replace(" ", "")
            simplified_original = original_text.replace(" ", "")
            found_pos = simplified_original.find(simplified_frag, current_global_pos)
            if found_pos != -1:
                non_space_count = original_text[:found_pos].count(" ")
                found_pos += non_space_count

        if found_pos == -1:
            if frag_idx in positions:
                calibrated_positions[frag_idx] = positions[frag_idx]
            else:
                calibrated_positions[frag_idx] = (
                    current_global_pos,
                    current_global_pos + frag_len,
                )
            continue

        calibrated_positions[frag_idx] = (found_pos, found_pos + frag_len)
        current_global_pos = found_pos + frag_len

    return calibrated_positions


def cut_text_by_semantic_length(text, max_semantic_length=50, min_tail_length=5):
    if not has_valid_content(text):
        return {"fragments": [], "positions": {}}

    original_text = text
    DOT_PLACEHOLDER = "##DOT##"

    processed_text = re.sub(r"(\d)\.(\d)", r"\1" + DOT_PLACEHOLDER + r"\2", text)
    for _ in range(3):
        processed_text = re.sub(
            r"([A-Z])\.([A-Z])", r"\1" + DOT_PLACEHOLDER + r"\2", processed_text
        )

    processed_text = processed_text.replace("\n", " ").replace("\u3002\uff0c", "\u3002")

    if get_semantic_length(processed_text) <= max_semantic_length:
        return {
            "fragments": [processed_text.replace(DOT_PLACEHOLDER, ".")],
            "positions": {0: (0, len(text))},
        }

    normalized_text = (
        processed_text.replace(".", "\u3002")
        .replace("!", "\uff01")
        .replace("?", "\uff1f")
        .replace(",", "\uff0c")
    )

    result_fragments = []
    position_map = {}

    sentences = []
    sentence_positions = []
    current_sentence = ""
    start_idx = 0

    for i, char in enumerate(normalized_text):
        current_sentence += char
        if char in "\u3002\uff01\uff1f":
            sentence = current_sentence.strip()
            if sentence:
                sentences.append(sentence)
                sentence_positions.append((start_idx, i + 1))
            current_sentence = ""
            start_idx = i + 1

    if current_sentence:
        sentences.append(current_sentence.strip())
        sentence_positions.append((start_idx, len(normalized_text)))
        if not sentences[-1].endswith(("\u3002", "\uff01", "\uff1f")):
            sentences[-1] += "\u3002"

    fragment_counter = 0
    for sent_idx, (sentence, (sent_start, sent_end)) in enumerate(
        zip(sentences, sentence_positions)
    ):
        clauses = []
        clause_positions = []
        current_clause = ""
        clause_start = 0

        for i, char in enumerate(sentence):
            current_clause += char
            if char in "\uff0c;\uff1b":
                clause = current_clause.strip()
                if clause and has_valid_content(clause):
                    clauses.append(clause)
                    clause_positions.append((clause_start, i + 1))
                elif clause and clauses:
                    clauses[-1] += clause
                    clause_positions[-1] = (clause_positions[-1][0], i + 1)
                current_clause = ""
                clause_start = i + 1

        if current_clause:
            clause = current_clause.strip()
            if clause and has_valid_content(clause):
                clauses.append(clause)
                clause_positions.append((clause_start, len(sentence)))
            elif clause and clauses:
                clauses[-1] += clause
                clause_positions[-1] = (clause_positions[-1][0], len(sentence))

        i = 0
        while i < len(clauses):
            clause = clauses[i]
            clause_start_in_sent, clause_end_in_sent = clause_positions[i]
            abs_start = sent_start + clause_start_in_sent
            abs_end = sent_start + clause_end_in_sent
            clause_semantic_len = get_semantic_length(clause)

            if clause_semantic_len < min_tail_length and i + 1 < len(clauses):
                next_clause = clauses[i + 1]
                next_start, next_end = clause_positions[i + 1]
                combined_clause = clause + next_clause
                combined_semantic_len = get_semantic_length(combined_clause)
                if combined_semantic_len <= max_semantic_length:
                    result_fragments = append_text_fragment(
                        result_fragments,
                        combined_clause,
                        max_semantic_length,
                        min_tail_length,
                    )
                    merged_position = (abs_start, sent_start + next_end)
                    position_map[fragment_counter] = merged_position
                    fragment_counter += 1
                    i += 2
                    continue

            if clause_semantic_len > max_semantic_length:
                sub_fragments = split_long_fragment(clause, max_semantic_length)
                frag_len = len(clause)
                sub_frag_count = len(sub_fragments)
                for j, frag in enumerate(sub_fragments):
                    result_fragments = append_text_fragment(
                        result_fragments, frag, max_semantic_length, min_tail_length
                    )
                    frag_len_part = len(frag)
                    frag_start = abs_start + j * (frag_len // sub_frag_count)
                    frag_end = frag_start + frag_len_part
                    position_map[fragment_counter] = (frag_start, frag_end)
                    fragment_counter += 1
            else:
                result_fragments = append_text_fragment(
                    result_fragments, clause, max_semantic_length, min_tail_length
                )
                position_map[fragment_counter] = (abs_start, abs_end)
                fragment_counter += 1
            i += 1

    final_result = [frag.replace(DOT_PLACEHOLDER, ".") for frag in result_fragments]
    calibrated_positions = calibrate_positions(
        final_result, position_map, original_text
    )
    return {"fragments": final_result, "positions": calibrated_positions}
