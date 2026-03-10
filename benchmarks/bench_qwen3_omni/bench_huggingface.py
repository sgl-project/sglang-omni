"""
Benchmark Qwen3-Omni with Hugging Face.

Usage:
# image only
python bench_huggingface.py --image-path ../../tests/data/cars.jpg

# video only
python bench_huggingface.py --video-path ../../tests/data/draw.mp4

# audio only
python bench_huggingface.py --audio-path ../../tests/data/cough.wav

# all modalities
python bench_huggingface.py --image-path ../../tests/data/cars.jpg --video-path ../../tests/data/draw.mp4 --audio-path ../../tests/data/cough.wav

Options:
  --img-path TEXT        Path to the image file
  --video-path TEXT      Path to the video file
  --audio-path TEXT      Path to the audio file
"""

import argparse
import time

import numpy as np
import torch
from transformers import (
    GenerationConfig,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    StoppingCriteriaList,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def get_timestamp():
    torch.cuda.synchronize()
    return time.time()


class Timer:

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    @property
    def elapsed_time(self):
        return self.end_time - self.start_time

    def __enter__(self):
        self.start_time = get_timestamp()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = get_timestamp()


def build_thinker_and_processor(model_path: str):
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_path,
        dtype="bfloat16",
        device_map="cuda",
        attn_implementation="sdpa",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    return model, processor


def main():
    model, processor = build_thinker_and_processor("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    args = parse_args()
    content = []

    if args.image_path:
        content.append({"type": "image", "image": args.image_path})
    if args.video_path:
        content.append({"type": "video", "video": args.video_path})
    if args.audio_path:
        content.append({"type": "audio", "audio": args.audio_path})

    end_to_end_time_list = []
    generated_token_count_list = []

    conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                *content,
                {
                    "type": "text",
                    "text": "Can you describe the content of these files in detail?",
                },
            ],
        },
    ]

    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(
        EosTokenCriteria(eos_token_id=processor.tokenizer.eos_token_id)
    )

    for idx in range(5):
        with Timer("end-to-end") as end_to_end_timer:
            with Timer("preprocessing") as processing_timer:
                # build conversations and run preprocessing
                inputs = processor.apply_chat_template(
                    conversations,
                    load_audio_from_video=False,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    fps=1,
                    # kwargs to be passed to `Qwen3OmniMoeProcessor`
                    padding=True,
                    use_audio_in_video=False,
                ).to(model.device)

            with Timer("generation") as generation_timer:
                if args.audio_path:
                    inputs["input_features"] = inputs["input_features"].to(
                        torch.bfloat16
                    )
                text_ids = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    generation_config=GenerationConfig(
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature
                    ),
                )
                input_ids_length = inputs["input_ids"].shape[1]
                response_ids = text_ids[:, input_ids_length:]
                response_length = response_ids.shape[1]
                text = processor.batch_decode(
                    response_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

        processing_latency = processing_timer.elapsed_time
        generation_latency = generation_timer.elapsed_time
        end_to_end_latency = end_to_end_timer.elapsed_time
        end_to_end_throughput = (response_length) / end_to_end_latency
        generation_throughput = response_length / generation_latency

        print("=" * 10, f"Iteration {idx + 1}", "=" * 10)
        print(f"Response: {text[0]}")
        print(f"Number of generated tokens: {response_length}")
        print(f"Processing latency: {processing_latency} seconds")
        print(f"Generation latency: {generation_latency} seconds")
        print(f"Generation throughput: {generation_throughput} tokens/second")
        print(f"End-to-end latency: {end_to_end_latency} seconds")
        print(f"End-to-end throughput: {end_to_end_throughput} tokens/second")
        print("\n\n")

        end_to_end_time_list.append(end_to_end_latency)
        generated_token_count_list.append(response_length)

    print(f"Average stats for the last 3 iterations:")
    avg_end_to_end_time = np.mean(end_to_end_time_list[-3:])
    avg_generated_token_count = np.mean(generated_token_count_list[-3:])
    print(f"End-to-end time: {avg_end_to_end_time} seconds")
    print(f"Generated token count: {avg_generated_token_count}")
    print(
        f"End-to-end throughput: {avg_generated_token_count / avg_end_to_end_time} tokens/second"
    )


if __name__ == "__main__":
    main()
