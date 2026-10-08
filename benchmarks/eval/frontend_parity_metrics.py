"""Summarize frozen frontend comparisons without discarding raw transcripts."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from jiwer import cer, wer


def normalize_transcript(text: str, *, chinese: bool) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith(("P", "S"))
    )
    if chinese:
        return "".join(text.split())
    else:
        return " ".join(text.split())


def choice_answer(text: str) -> str | None:
    text = text.strip()
    direct = re.fullmatch(r"[\s*()]*([A-D])[\s*().]*", text)
    if direct:
        return direct.group(1)
    else:
        answers = re.findall(
            r"(?:answer\s*(?:is|:)|correct\s+(?:answer|option)\s*(?:is|:))\s*[*\s(]*([A-D])\b",
            text,
            flags=re.IGNORECASE,
        )
        if answers:
            return answers[-1].upper()
        else:
            return None


def summarize(result_path: Path) -> dict[str, dict[str, int | float | None]]:
    results = json.loads(result_path.read_text())
    summary: dict[str, dict[str, int | float | None]] = {}
    for task in sorted({sample["task"] for sample in results}):
        samples = [sample for sample in results if sample["task"] == task]
        errors = sum("error" in sample for sample in samples)
        if task.startswith("asr"):
            chinese = task == "asr_zh"
            references = [
                normalize_transcript(sample["reference"], chinese=chinese)
                for sample in samples
                if "error" not in sample
            ]
            hypotheses = [
                normalize_transcript(sample["text"], chinese=chinese)
                for sample in samples
                if "error" not in sample
            ]
            summary[task] = {
                "count": len(samples),
                "errors": errors,
                "cer" if chinese else "wer": (
                    (
                        cer(references, hypotheses)
                        if chinese
                        else wer(references, hypotheses)
                    )
                    if references
                    else None
                ),
            }
            extracted_hypotheses = [
                normalize_transcript(
                    sample["text"].partition("|||")[0], chinese=chinese
                )
                for sample in samples
                if "error" not in sample
            ]
            summary[task]["metadata_suffix_count"] = sum(
                "|||" in sample.get("text", "") for sample in samples
            )
            summary[task]["extracted_cer" if chinese else "extracted_wer"] = (
                (
                    cer(references, extracted_hypotheses)
                    if chinese
                    else wer(references, extracted_hypotheses)
                )
                if references
                else None
            )
        else:
            answers = [choice_answer(sample.get("text", "")) for sample in samples]
            correct = sum(
                answer is not None
                and "error" not in sample
                and answer == sample["reference"]
                for answer, sample in zip(answers, samples)
            )
            summary[task] = {
                "count": len(samples),
                "errors": errors,
                "missing_answer": sum(answer is None for answer in answers),
                "correct": correct,
                "accuracy": correct / len(samples),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    summary = {result.name: summarize(result) for result in arguments.results}
    rendered = json.dumps(summary, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered)
    else:
        pass
    print(rendered)


if __name__ == "__main__":
    main()
