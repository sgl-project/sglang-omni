"""Run frozen multimodal correctness fixtures through HF or an Omni API."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import httpx
import librosa
import torch
import transformers
from datasets import Audio
from datasets import Image as DatasetImage
from datasets import load_dataset
from huggingface_hub import HfApi
from minicpmo.utils import get_video_frame_audio_segments
from PIL import Image
from transformers import AutoModel


def prepare_video(output: Path, dataset_path: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for row in map(
        json.loads, (dataset_path / "data/test.jsonl").read_text().splitlines()
    ):
        if row["duration"] != "short":
            continue
        else:
            media_path = (dataset_path / row["video_path"]).resolve()
            samples.append(
                {
                    "id": f"video-{row['question_id']}",
                    "task": "video",
                    "reference": row["answer"],
                    "answer": row["answer"],
                    "media_sha256": [
                        hashlib.sha256(media_path.read_bytes()).hexdigest()
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": str(media_path),
                                        "use_audio": True,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": row["question"]
                                    + "\n"
                                    + "\n".join(row["options"])
                                    + "\nAnswer with one letter only.",
                                },
                            ],
                        }
                    ],
                }
            )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "zhaochenyang20/Video_MME_ci": "833bd815c628ff277911bea3b1563545b21d5e27"
                },
                "samples": samples[:100],
            },
            indent=2,
        )
        + "\n"
    )


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    sources = {}
    for dataset_name, split, task in [
        ("zhaochenyang20/mmmu-ci-50", "validation", "mmmu"),
        ("zhaochenyang20/mmsu-ci-2000", "train", "mmsu"),
        ("zhaochenyang20/seed-tts-eval-arrow", "en", "asr_en"),
        ("zhaochenyang20/seed-tts-eval-arrow", "zh", "asr_zh"),
    ]:
        revision = HfApi().dataset_info(dataset_name).sha
        sources[dataset_name] = revision
        dataset = load_dataset(dataset_name, split=split, revision=revision)
        dataset = dataset.select(range(min(100, len(dataset))))
        if task == "mmsu":
            dataset = dataset.cast_column("audio", Audio(decode=False))
        elif task.startswith("asr"):
            dataset = dataset.cast_column("ref_audio", Audio(decode=False))
        else:
            for column in dataset.column_names:
                if column.startswith("image_"):
                    dataset = dataset.cast_column(column, DatasetImage(decode=False))
        for index, row in enumerate(dataset):
            sample_id = f"{task}-{index:04d}"
            parts = []
            if task == "mmsu":
                media_path = output / f"{sample_id}.mp3"
                media_path.write_bytes(row["audio"]["bytes"])
                choices = [str(row[f"choice_{letter}"]) for letter in "abcd"]
                prompt = (
                    str(row["question"])
                    + "\n"
                    + "\n".join(
                        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
                    )
                    + "\nAnswer with one letter only."
                )
                answer = next(
                    (
                        letter
                        for letter, choice in zip("ABCD", choices)
                        if choice.strip() == str(row["answer_gt"]).strip()
                    ),
                    str(row["answer_gt"]),
                )
                parts = [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": str(media_path.resolve())},
                    },
                    {"type": "text", "text": prompt},
                ]
            elif task.startswith("asr"):
                media_path = output / f"{sample_id}.wav"
                media_path.write_bytes(row["ref_audio"]["bytes"])
                prompt = (
                    "Please listen to the audio snippet carefully and transcribe the content."
                    if task == "asr_en"
                    else "请仔细听这段音频片段，并将其内容逐字记录。"
                )
                answer = row["ref_text"]
                parts = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": str(media_path.resolve())},
                    },
                ]
            else:
                question = str(row["question"])
                options = row["options"]
                if isinstance(options, str):
                    options = ast.literal_eval(options)
                prompt = (
                    question
                    + "\n"
                    + "\n".join(
                        f"{chr(65+i)}. {choice}" for i, choice in enumerate(options)
                    )
                    + "\nAnswer with one letter only."
                )
                for image_index in range(1, 8):
                    image = row.get(f"image_{image_index}")
                    if image:
                        media_path = output / f"{sample_id}-{image_index}.png"
                        if image.get("bytes"):
                            media_path.write_bytes(image["bytes"])
                        else:
                            Image.open(image["path"]).save(media_path)
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": str(media_path.resolve())},
                            }
                        )
                parts.append({"type": "text", "text": prompt})
                answer = row["answer"]
            manifest.append(
                {
                    "id": sample_id,
                    "source_id": str(row.get("id", row.get("sample_id", index))),
                    "task": task,
                    "answer": answer,
                    "messages": [{"role": "user", "content": parts}],
                }
            )
    for sample in manifest:
        sample["media_sha256"] = [
            hashlib.sha256(Path(part[part["type"]]["url"]).read_bytes()).hexdigest()
            for message in sample["messages"]
            for part in message["content"]
            if part["type"] != "text"
        ]
    (output / "manifest.json").write_text(
        json.dumps(
            {"sources": sources, "samples": manifest}, ensure_ascii=False, indent=2
        )
        + "\n"
    )
    print(f"Prepared {len(manifest)} fixtures")


def native_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for message in messages:
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append(part["text"])
            elif part["type"] == "image_url":
                content.append(Image.open(part["image_url"]["url"]).convert("RGB"))
            elif part["type"] == "video_url":
                frames, segments, _ = get_video_frame_audio_segments(
                    part["video_url"]["url"]
                )
                content.extend(
                    element for pair in zip(frames, segments) for element in pair
                )
            else:
                waveform, _ = librosa.load(
                    part["audio_url"]["url"], sr=16000, mono=True
                )
                content.append(waveform)
        result.append({"role": message["role"], "content": content})
    return result


def run(arguments: argparse.Namespace) -> None:
    manifest = json.loads(arguments.manifest.read_text())
    samples = manifest["samples"]
    if arguments.limit:
        samples = samples[: arguments.limit]
    results = []
    if arguments.resume and arguments.output.exists():
        results = [
            sample
            for sample in json.loads(arguments.output.read_text())
            if "error" not in sample
        ]
        completed_ids = {sample["id"] for sample in results}
        samples = [sample for sample in samples if sample["id"] not in completed_ids]
    else:
        pass
    model = None
    processor = None
    if arguments.backend == "official":
        if arguments.family == "qwen":
            model = (
                transformers.Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    arguments.model_path,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                .eval()
                .cuda()
            )
            model.disable_talker()
            processor = transformers.Qwen3OmniMoeProcessor.from_pretrained(
                arguments.model_path
            )
        else:
            model = (
                AutoModel.from_pretrained(
                    arguments.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    init_tts=False,
                    attn_implementation="sdpa",
                )
                .eval()
                .cuda()
            )
    with httpx.Client(timeout=600) as client:
        for sample in samples:
            try:
                if processor is not None:
                    messages = []
                    images, audios = [], []
                    for message in native_messages(sample["messages"]):
                        parts = []
                        for part in message["content"]:
                            if isinstance(part, Image.Image):
                                images.append(part)
                                parts.append({"type": "image"})
                            elif isinstance(part, str):
                                parts.append({"type": "text", "text": part})
                            else:
                                audios.append(part)
                                parts.append({"type": "audio"})
                        messages.append({"role": message["role"], "content": parts})
                    prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    model_inputs = processor(
                        text=prompt,
                        images=images or None,
                        audio=audios or None,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(model.device, model.dtype)
                    with torch.inference_mode():
                        outputs = model.generate(
                            **model_inputs,
                            return_audio=False,
                            thinker_do_sample=False,
                            thinker_max_new_tokens=256,
                            thinker_repetition_penalty=1.0,
                        )
                    answer = processor.batch_decode(
                        outputs[:, model_inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                elif model is not None:
                    has_video = any(
                        part["type"] == "video_url"
                        for message in sample["messages"]
                        for part in message["content"]
                    )
                    answer = model.chat(
                        msgs=native_messages(sample["messages"]),
                        do_sample=False,
                        num_beams=1,
                        repetition_penalty=1.0,
                        max_new_tokens=256,
                        enable_thinking=False,
                        use_tts_template=False,
                        **(
                            {
                                "omni_mode": True,
                                "max_slice_nums": 1,
                                "use_image_id": False,
                            }
                            if has_video
                            else {}
                        ),
                    )
                else:
                    request = {
                        "model": "minicpm",
                        "messages": sample["messages"],
                        "temperature": 0,
                        "max_tokens": 256,
                        "seed": 0,
                        "modalities": ["text"],
                        "use_audio_in_video": any(
                            part["type"] == "video_url"
                            for message in sample["messages"]
                            for part in message["content"]
                        ),
                    }
                    if arguments.input_style == "top-level":
                        messages = []
                        for message in sample["messages"]:
                            texts = []
                            for part in message["content"]:
                                if part["type"] == "text":
                                    texts.append(part["text"])
                                else:
                                    name = (
                                        "images"
                                        if part["type"] == "image_url"
                                        else "audios"
                                    )
                                    request.setdefault(name, []).append(
                                        part[part["type"]]["url"]
                                    )
                            messages.append(
                                {"role": message["role"], "content": "\n".join(texts)}
                            )
                        request["messages"] = messages
                    response = client.post(
                        arguments.api_url + "/v1/chat/completions", json=request
                    )
                    response.raise_for_status()
                    answer = response.json()["choices"][0]["message"]["content"]
                results.append(
                    {
                        "id": sample["id"],
                        "task": sample["task"],
                        "reference": sample["answer"],
                        "text": answer,
                    }
                )
            except (ValueError, RuntimeError, httpx.HTTPError) as error:
                results.append(
                    {"id": sample["id"], "task": sample["task"], "error": str(error)}
                )
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n"
            )
            print(
                f"{sample['id']}: {results[-1].get('text', results[-1].get('error'))}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--video-dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--backend", choices=["official", "omni"])
    parser.add_argument("--family", choices=["minicpm", "qwen"], default="minicpm")
    parser.add_argument("--model-path")
    parser.add_argument("--api-url", default="http://127.0.0.1:18000")
    parser.add_argument(
        "--input-style", choices=["inline", "top-level"], default="inline"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.prepare:
        if arguments.video_dataset:
            prepare_video(arguments.prepare, arguments.video_dataset)
        else:
            prepare(arguments.prepare)
    else:
        run(arguments)


if __name__ == "__main__":
    main()
