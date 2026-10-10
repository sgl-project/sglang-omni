"""Run the unmodified Transformers model against prepared PCM16 audio."""

import argparse
import json
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import transformers
from numpy.typing import NDArray
from transformers import AutoModelForRNNT, AutoProcessor, Nemotron3_5AsrProcessor


def streaming_features(
    waveform: NDArray[np.float32], processor: Nemotron3_5AsrProcessor, language: str
) -> Iterator[torch.Tensor]:
    first_chunk_samples = processor.num_samples_first_audio_chunk
    subsequent_chunk_samples = processor.num_samples_per_audio_chunk
    first_chunk_frames = processor.num_mel_frames_first_audio_chunk
    subsequent_chunk_frames = processor.num_mel_frames_per_audio_chunk
    hop_samples = processor.feature_extractor.hop_length
    transform_samples = processor.feature_extractor.n_fft
    sample_count = len(waveform)
    first = np.pad(
        waveform[:first_chunk_samples], (0, max(0, first_chunk_samples - sample_count))
    )
    inputs = processor(
        first,
        sampling_rate=16000,
        language=language,
        is_streaming=True,
        is_first_audio_chunk=True,
        return_tensors="pt",
    )
    yield inputs.input_features[:, :first_chunk_frames, :]
    covered_samples = first_chunk_samples
    frame_index = first_chunk_frames
    while covered_samples < sample_count:
        start_sample = frame_index * hop_samples - transform_samples // 2
        end_sample = start_sample + subsequent_chunk_samples
        source_samples = waveform[max(0, start_sample) : min(end_sample, sample_count)]
        left_padding_samples = max(0, -start_sample)
        chunk = np.pad(
            source_samples,
            (
                left_padding_samples,
                subsequent_chunk_samples - left_padding_samples - len(source_samples),
            ),
        )
        inputs = processor(
            chunk,
            sampling_rate=16000,
            language=language,
            is_streaming=True,
            is_first_audio_chunk=False,
            return_tensors="pt",
        )
        yield inputs.input_features
        frame_index += subsequent_chunk_frames
        covered_samples = end_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["offline", "streaming"],
        default=["offline", "streaming"],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    if transformers.__version__ != "5.13.0":
        parser.error("Use a separate environment with transformers==5.13.0")
    elif arguments.repeats <= 0:
        parser.error("Repetitions must be positive")
    else:
        pass
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(
        arguments.model_path, local_files_only=True
    )
    processor.set_num_lookahead_tokens(arguments.lookahead)
    model = (
        AutoModelForRNNT.from_pretrained(
            arguments.model_path,
            dtype=torch.float32,
            local_files_only=True,
        )
        .to(arguments.device)
        .eval()
    )
    samples: list[tuple[str, str, NDArray[np.float32]]] = []
    for line in arguments.meta.read_text().splitlines():
        sample_id, reference, filename, target_text = line.split("|")
        with wave.open(str(arguments.meta.parent / filename), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
                16000,
                1,
                2,
            ):
                raise ValueError(f"Expected mono 16 kHz PCM16: {filename}")
            else:
                pass
            waveform = (
                np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768.0
            )
        samples.append((sample_id, reference, waveform))
    if not samples or len(
        {sample_id for sample_id, reference, waveform in samples}
    ) != len(samples):
        parser.error("Require nonempty metadata with unique sample identifiers")
    else:
        pass
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as output:
        for repeat in range(arguments.repeats):
            for mode in arguments.modes:
                for sample_id, reference, waveform in samples:
                    started_seconds = time.perf_counter()
                    inputs = processor(
                        waveform,
                        sampling_rate=16000,
                        language=arguments.language,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device, dtype=model.dtype)
                    if mode == "streaming":
                        features = streaming_features(
                            waveform, processor, arguments.language
                        )
                        generated_features = (
                            chunk.to(model.device, dtype=model.dtype)
                            for chunk in features
                        )
                        with torch.inference_mode():
                            result = model.generate(
                                input_features=generated_features,
                                prompt_ids=inputs.prompt_ids,
                                num_lookahead_tokens=arguments.lookahead,
                                return_dict_in_generate=True,
                            )
                    else:
                        with torch.inference_mode():
                            result = model.generate(
                                **inputs, return_dict_in_generate=True
                            )
                    tokens = result.sequences.cpu().tolist()
                    record = {
                        "sample_id": sample_id,
                        "model_path": arguments.model_path,
                        "device": arguments.device,
                        "reference": reference,
                        "mode": mode,
                        "repeat": repeat,
                        "language": arguments.language,
                        "lookahead": arguments.lookahead,
                        "tokens": tokens,
                        "text": processor.batch_decode(
                            result.sequences, skip_special_tokens=True
                        )[0],
                        "raw_text": processor.batch_decode(
                            result.sequences, skip_special_tokens=False
                        )[0],
                        "diagnostic_wall_seconds": time.perf_counter()
                        - started_seconds,
                        "transformers": transformers.__version__,
                        "torch": torch.__version__,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        f"{mode} repeat={repeat} sample={sample_id}: {record['text']}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
else:
    pass
