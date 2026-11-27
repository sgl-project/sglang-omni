# SPDX-License-Identifier: Apache-2.0
"""Load NVIDIA's Nemotron Unified ASR ``.nemo`` release without NeMo.

The archive's config, SentencePiece tokenizer and weights become a Transformers
``Nemotron3_5AsrForRNNT`` with a ``parakeet_encoder`` and its processor.
"""

from __future__ import annotations

import re
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml
from tokenizers import AddedToken
from transformers import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrForRNNT,
    Nemotron3_5AsrProcessor,
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
    ParakeetTokenizer,
)
from transformers.convert_slow_tokenizer import ParakeetConverter

# NeMo parameter names to the Transformers Parakeet and Nemotron 3.5 ASR layout,
# following the Transformers converters for those models.
NEMO_PARAMETER_RENAMES = {
    r"^prompt_kernel\.0\.": "prompt_projector.linear_1.",
    r"^prompt_kernel\.2\.": "prompt_projector.linear_2.",
    r"encoder\.pre_encode\.conv\.": "encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": "encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": "encoder.encode_positions.",
    r"\.conv\.batch_norm\.": ".conv.norm.",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": "o_proj",
    r"linear_q": "q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": "relative_k_proj",
    r"decoder\.prediction\.embed\.": "decoder.embedding.",
    r"decoder\.prediction\.dec_rnn\.lstm\.": "decoder.lstm.",
    r"joint\.enc\.": "encoder_projector.",
    r"joint\.pred\.": "decoder.decoder_projector.",
    r"joint\.joint_net\.2\.": "joint.head.",
}


def load_nemo_archive(
    path: Path, dtype: torch.dtype
) -> tuple[Nemotron3_5AsrProcessor, Nemotron3_5AsrForRNNT]:
    """Build the Transformers processor and model from NVIDIA's .nemo release."""
    with tarfile.open(path) as archive, tempfile.TemporaryDirectory() as scratch:
        members = {Path(m.name).name: m for m in archive if m.isfile()}
        config = yaml.safe_load(archive.extractfile(members["model_config.yaml"]))
        tokenizer_model = Path(scratch, "tokenizer.model")
        tokenizer_member = config["tokenizer"]["model_path"].removeprefix("nemo:")
        tokenizer_model.write_bytes(
            archive.extractfile(members[tokenizer_member]).read()
        )
        weights = torch.load(
            archive.extractfile(members["model_weights.ckpt"]),
            map_location="cpu",
            weights_only=True,
        )
        tokenizer_object = ParakeetConverter(str(tokenizer_model)).converted()
    blank = len(config["labels"])
    prompts = {
        str(k): int(v) for k, v in config["model_defaults"]["prompt_dictionary"].items()
    }
    state_dict = {}
    for name, tensor in weights.items():
        for pattern, replacement in NEMO_PARAMETER_RENAMES.items():
            name = re.sub(pattern, replacement, name)
        if "featurizer" not in name:
            state_dict[name] = tensor
        else:
            pass
    encoder, prednet = config["encoder"], config["decoder"]["prednet"]
    model_config = Nemotron3_5AsrConfig(
        vocab_size=blank + 1,
        blank_token_id=blank,
        pad_token_id=blank,
        decoder_hidden_size=prednet["pred_hidden"],
        num_decoder_layers=prednet["pred_rnn_layers"],
        hidden_act=config["joint"]["jointnet"]["activation"],
        max_symbols_per_step=config["decoding"]["greedy"]["max_symbols"],
        num_prompts=config["model_defaults"]["num_prompts"],
        prompt_intermediate_size=state_dict["prompt_projector.linear_1.weight"].shape[
            0
        ],
        default_prompt_id=prompts["auto"],
        encoder_config=ParakeetEncoderConfig(
            hidden_size=encoder["d_model"],
            num_hidden_layers=encoder["n_layers"],
            num_attention_heads=encoder["n_heads"],
            intermediate_size=encoder["d_model"] * encoder["ff_expansion_factor"],
            conv_kernel_size=encoder["conv_kernel_size"],
            subsampling_factor=encoder["subsampling_factor"],
            subsampling_conv_channels=encoder["subsampling_conv_channels"],
            num_mel_bins=encoder["feat_in"],
            max_position_embeddings=encoder["pos_emb_max_len"],
            scale_input=encoder["xscaling"],
        ).to_dict(),
    )
    model = Nemotron3_5AsrForRNNT(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.generation_config.decoder_start_token_id = blank
    special = {
        name: AddedToken(token, normalized=False, special=True)
        for name, token in (("pad_token", "<blank>"), ("unk_token", "<unk>"))
    }
    tokenizer = ParakeetTokenizer(
        tokenizer_object=tokenizer_object, clean_up_tokenization_spaces=False
    )
    tokenizer.add_special_tokens(special)
    if tokenizer.pad_token_id != blank:
        raise ValueError(f"Nemotron .nemo tokenizer has no blank at id {blank}")
    else:
        pass
    features = config["preprocessor"]
    rate = features["sample_rate"]
    processor = Nemotron3_5AsrProcessor(
        feature_extractor=ParakeetFeatureExtractor(
            feature_size=features["features"],
            sampling_rate=rate,
            hop_length=int(features["window_stride"] * rate),
            win_length=int(features["window_size"] * rate),
            n_fft=features["n_fft"],
        ),
        tokenizer=tokenizer,
        prompt_dictionary=prompts,
        num_prompts=model_config.num_prompts,
        supported_num_lookahead_tokens=[0],
        default_num_lookahead_tokens=0,
    )
    return processor, model.to(dtype)


__all__ = ["load_nemo_archive"]
