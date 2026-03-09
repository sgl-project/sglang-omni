# SPDX-License-Identifier: Apache-2.0
"""Capture official HF parity artifacts for the canonical speech prompt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.qwen3_omni_parity.common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_PROMPT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEED,
    DEFAULT_SPEAKER,
    add_repo_root_to_syspath,
    default_model_path,
    file_sha256,
    load_hf_full_model,
    prompt_hash,
    save_json,
)

add_repo_root_to_syspath()

from sglang_omni.client.audio import encode_wav


def build_talker_prefill(
    model: Any,
    thinker_result: Any,
    input_ids: torch.Tensor,
    thinker_embed: torch.Tensor,
    thinker_hidden: torch.Tensor,
    *,
    speaker: str,
) -> dict[str, torch.Tensor]:
    im_start_indexes = torch.cat(
        (
            torch.nonzero(input_ids[0] == model.config.im_start_token_id).squeeze(),
            torch.tensor(
                [thinker_result.sequences.shape[-1]],
                device=input_ids.device,
                dtype=input_ids.dtype,
            ),
        ),
        dim=-1,
    ).to(model.talker.device)
    multimodal_mask = (
        (thinker_result.sequences == model.config.thinker_config.audio_token_id)
        | (thinker_result.sequences == model.config.thinker_config.image_token_id)
        | (thinker_result.sequences == model.config.thinker_config.video_token_id)
    ).to(model.talker.device)

    talker_special_tokens = torch.tensor(
        [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
        device=model.thinker.device,
        dtype=input_ids.dtype,
    )
    tts_bos_embed, tts_eos_embed, tts_pad_embed = (
        model.talker.text_projection(model.thinker.get_input_embeddings()(talker_special_tokens))
        .to(model.talker.device)
        .chunk(3, dim=1)
    )
    speaker_id = model.config.talker_config.speaker_id[speaker.lower()]

    talker_input_embeds = []
    talker_input_ids = []
    trailing_text_hidden = None
    for i in range(len(im_start_indexes) - 1):
        im_start_index = im_start_indexes[i]
        segment_end_index = im_start_indexes[i + 1]
        role_token = input_ids[0][im_start_index + 1]
        if role_token == model.config.system_token_id:
            continue
        if role_token == model.config.user_token_id:
            talker_user_part = model._get_talker_user_parts(
                im_start_index,
                segment_end_index,
                multimodal_mask,
                thinker_hidden,
                thinker_embed,
            )
            talker_input_embeds.append(talker_user_part)
            talker_input_ids.append(
                thinker_result.sequences[:, im_start_index:segment_end_index]
            )
        elif role_token == model.config.assistant_token_id and i == len(im_start_indexes) - 2:
            (
                talker_assistant_embeds,
                talker_assistant_ids,
                trailing_text_hidden,
            ) = model._get_talker_assistant_parts(
                im_start_index,
                segment_end_index,
                speaker_id,
                thinker_embed,
                tts_pad_embed,
                tts_bos_embed,
                tts_eos_embed,
            )
            talker_input_embeds.append(talker_assistant_embeds)
            talker_input_ids.append(talker_assistant_ids)

    return {
        "input_embeds": torch.cat(talker_input_embeds, dim=1).detach().cpu()[0],
        "input_ids": torch.cat(talker_input_ids, dim=1).detach().cpu()[0],
        "trailing_text_hidden": (
            trailing_text_hidden.detach().cpu()[0]
            if trailing_text_hidden is not None
            else None
        ),
        "tts_pad_embed": tts_pad_embed.detach().cpu()[0, 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--thinker-max-new-tokens", type=int, default=48)
    parser.add_argument("--talker-max-new-tokens", type=int, default=256)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--capture-predictor", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_id = prompt_hash(args.prompt)

    processor, model = load_hf_full_model(
        args.model_path, device=args.device, dtype=torch.bfloat16
    )

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    predictor_calls: list[dict[str, object]] = []
    orig_cp_generate = model.talker.code_predictor.generate

    def wrapped_cp_generate(*g_args, **g_kwargs):
        call_index = len(predictor_calls)
        input_embeds = g_kwargs.get("inputs_embeds")
        if input_embeds is None and g_args:
            input_embeds = g_args[0]
        record: dict[str, object] = {"call_index": call_index}
        if isinstance(input_embeds, torch.Tensor):
            record["inputs_embeds_shape"] = list(input_embeds.shape)
            record["talker_hidden"] = input_embeds[0, 0].detach().cpu().float().tolist()
            record["layer0_embed"] = input_embeds[0, 1].detach().cpu().float().tolist()
        out = orig_cp_generate(*g_args, **g_kwargs)
        record["sequences"] = out.sequences.detach().cpu().tolist()
        predictor_calls.append(record)
        return out

    model.talker.code_predictor.generate = wrapped_cp_generate

    with torch.no_grad():
        thinker_result = model.thinker.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=args.thinker_max_new_tokens,
            eos_token_id=151645,
            do_sample=False,
            temperature=0.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        thinker_embed = torch.cat(
            [hidden_states[0] for hidden_states in thinker_result.hidden_states], dim=1
        ).to(model.talker.device)
        thinker_hidden = torch.cat(
            [
                hidden_states[model.config.talker_config.accept_hidden_layer]
                for hidden_states in thinker_result.hidden_states
            ],
            dim=1,
        ).to(model.talker.device)

        talker_prefill = build_talker_prefill(
            model,
            thinker_result,
            inputs["input_ids"],
            thinker_embed,
            thinker_hidden,
            speaker=args.speaker,
        )
        talker_prefill_path = out_dir / f"hf_talker_prefill_{prompt_id}_seed{args.seed}.pt"
        torch.save(talker_prefill, talker_prefill_path)

        suppress_tokens = [
            token_id
            for token_id in range(
                max(model.config.talker_config.text_config.vocab_size - 1024, 0),
                model.config.talker_config.text_config.vocab_size,
            )
            if token_id != model.config.talker_config.codec_eos_token_id
        ]

        torch.manual_seed(args.seed)
        talker_result = model.talker.generate(
            inputs_embeds=talker_prefill["input_embeds"].unsqueeze(0).to(model.talker.device),
            trailing_text_hidden=(
                talker_prefill["trailing_text_hidden"].unsqueeze(0).to(model.talker.device)
                if isinstance(talker_prefill["trailing_text_hidden"], torch.Tensor)
                else None
            ),
            tts_pad_embed=talker_prefill["tts_pad_embed"].to(model.talker.device),
            talker_input_ids=talker_prefill["input_ids"].unsqueeze(0).to(model.talker.device),
            max_new_tokens=args.talker_max_new_tokens,
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            repetition_penalty=args.talker_repetition_penalty,
            eos_token_id=model.config.talker_config.codec_eos_token_id,
            suppress_tokens=suppress_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        layer0_codes = talker_result.sequences[0].detach().cpu()
        residual_rows = [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None]
        full_codes = torch.stack(residual_rows, dim=1).transpose(1, 2).detach().cpu()
        wav = model.code2wav.chunked_decode(full_codes.to(model.talker.device), chunk_size=300, left_context_size=25)

    generated_ids = thinker_result.sequences[0, prompt_len:].detach().cpu().tolist()
    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    audio = wav.reshape(-1).detach().cpu().float().numpy()
    wav_path = out_dir / f"hf_capture_{prompt_id}_seed{args.seed}.wav"
    wav_path.write_bytes(encode_wav(audio, sample_rate=DEFAULT_SAMPLE_RATE))

    predictor_path = None
    if args.capture_predictor:
        predictor_path = out_dir / f"hf_predictor_capture_{prompt_id}_seed{args.seed}.json"
        save_json(
            {
                "seed": args.seed,
                "prompt": args.prompt,
                "generated_ids": generated_ids,
                "generated_text": generated_text,
                "num_predictor_calls": len(predictor_calls),
                "predictor_calls": predictor_calls,
            },
            predictor_path,
        )

    summary = {
        "seed": args.seed,
        "prompt": args.prompt,
        "speaker": args.speaker,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "layer0_len": int(layer0_codes.numel()),
        "layer0_codes": layer0_codes.tolist(),
        "codec_codes_shape": list(full_codes.shape),
        "codec_codes": full_codes[0].tolist(),
        "codec_eos_token_id": int(model.config.talker_config.codec_eos_token_id),
        "wav_num_samples": int(audio.size),
        "wav_duration_sec": float(audio.size / DEFAULT_SAMPLE_RATE),
        "audio_path": str(wav_path),
        "audio_sha256": file_sha256(wav_path),
        "talker_prefill_path": str(talker_prefill_path),
        "predictor_capture_path": str(predictor_path) if predictor_path is not None else None,
        "trailing_len": (
            int(talker_prefill["trailing_text_hidden"].shape[0])
            if isinstance(talker_prefill["trailing_text_hidden"], torch.Tensor)
            else 0
        ),
    }
    save_json(summary, out_dir / f"hf_capture_{prompt_id}_seed{args.seed}.json")
    print(summary["layer0_len"])


if __name__ == "__main__":
    main()
