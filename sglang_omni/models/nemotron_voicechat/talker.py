from __future__ import annotations

import torch
from sglang.srt.models.gemma3_causal import Gemma3ForCausalLM
from sglang.srt.server_args import get_global_server_args
from torch import nn
from torch.nn import functional
from transformers import T5GemmaConfig, T5GemmaEncoderModel, T5GemmaModuleConfig

from sglang_omni.models.nemotron_voicechat.mog_head import MoGHead, RMSNorm


class SubwordFlagEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("pad_tensor", torch.tensor(vocab_size, dtype=torch.long))
        self.register_buffer(
            "is_continuation", torch.zeros(vocab_size + 1, dtype=torch.long)
        )
        self.cont_emb = nn.Embedding(2, hidden_size)

    def forward(self, embeds_TD, token_ids_T):
        safe_T = torch.where(
            token_ids_T >= self.vocab_size, self.pad_tensor, token_ids_T
        )
        return embeds_TD + self.cont_emb(self.is_continuation[safe_T])


class BosEosEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("pad_tensor", torch.tensor(vocab_size, dtype=torch.long))
        self.register_buffer("special_flags", torch.zeros(vocab_size, dtype=torch.long))
        self.special_emb = nn.Embedding(3, hidden_size)

    def forward(self, embeds_TD, token_ids_T):
        safe_T = torch.where(
            token_ids_T >= self.vocab_size, self.pad_tensor, token_ids_T
        )
        return embeds_TD + self.special_emb(self.special_flags[safe_T])


class GatedFusion(nn.Module):
    def __init__(self, hidden_size: int, num_codebooks: int) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.audio_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self.final_norm = RMSNorm(hidden_size)

    def forward(self, audio_TD, text_TD):
        audio_TD = self.audio_proj(audio_TD / self.num_codebooks)
        text_TD = self.text_proj(text_TD)
        gate_D = torch.sigmoid(self.gate.float()).to(audio_TD.dtype)
        scale = torch.sigmoid(self.residual_scale.float()).to(audio_TD.dtype)
        fused_TD = gate_D * audio_TD + (1.0 - gate_D) * text_TD
        return self.final_norm((scale * fused_TD).float()).to(audio_TD.dtype)


class TalkerEmbedding(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        vocab_size = int(config["vocab_size"])
        self.char_padding_idx = int(config["char_vocab_size"])
        self.embed_tokens = nn.Embedding(
            self.char_padding_idx + 1, hidden_size, padding_idx=self.char_padding_idx
        )
        self.backbone = build_char_encoder(config["char_encoder_config"])
        self.proj_embedding = nn.Linear(hidden_size, hidden_size, bias=False)
        self.subword_flag_emb = SubwordFlagEmbedding(vocab_size, hidden_size)
        self.bos_eos_emb = BosEosEmbedding(vocab_size, hidden_size)

    def forward(self, token_ids_T, char_ids_TC, char_lengths_T, pooled_mask_T=None):
        char_embeds_TCD = self.embed_tokens(char_ids_TC)
        positions_C = torch.arange(char_ids_TC.shape[1], device=char_ids_TC.device)
        valid_TC = positions_C[None, :] < char_lengths_T[:, None]
        hidden_TCD = self.backbone(
            inputs_embeds=char_embeds_TCD, attention_mask=valid_TC.long()
        ).last_hidden_state
        pooled_TD = (hidden_TCD * valid_TC[..., None]).sum(1) / char_lengths_T[
            :, None
        ].clamp_min(1)
        embeds_TD = self.proj_embedding(pooled_TD)
        if pooled_mask_T is not None:
            embeds_TD = embeds_TD * pooled_mask_T[:, None]
        embeds_TD = self.subword_flag_emb(embeds_TD, token_ids_T)
        return self.bos_eos_emb(embeds_TD, token_ids_T)


def build_char_encoder(config: dict) -> nn.Module:
    # Separate config objects: T5GemmaConfig marks the one it takes as the
    # decoder causal in place, which would make this encoder causal too.
    module_config = lambda: T5GemmaModuleConfig(**config["encoder"], vocab_size=1)
    model = T5GemmaEncoderModel(
        T5GemmaConfig(
            encoder=module_config(), decoder=module_config(), is_encoder_decoder=False
        )
    )
    model.set_input_embeddings(nn.Identity())
    return model


class EarTtsTalker(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.num_quantizers = int(config["num_quantizers"])
        self.embed_subword = TalkerEmbedding(config)
        self.gated_fusion_audio_text = GatedFusion(hidden_size, self.num_quantizers)
        self.bos_emb = nn.Parameter(torch.empty(hidden_size))
        self.null_emb = nn.Parameter(torch.empty(hidden_size))
        self.audio_prompt_projection_W = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.register_buffer(
            "rvq_embs",
            torch.empty(
                self.num_quantizers,
                int(config["codebook_size"]),
                int(config["latent_size"]),
            ),
        )
        self.embed_code = nn.Linear(int(config["latent_size"]), hidden_size, bias=False)

    def embed_codes(self, codes_TQ):
        return self.embed_code(self._depth_sum(codes_TQ, self.num_quantizers))

    def quantise(self, latent_TD, codes_TQ, first_level: int, count: int):
        residual_TD = latent_TD
        for level in range(first_level, first_level + count):
            codebook_CD = self.rvq_embs[level]
            score_TC = codebook_CD.pow(2).sum(-1) - 2.0 * (residual_TD @ codebook_CD.T)
            index_T = score_TC.argmin(-1)
            residual_TD = residual_TD - codebook_CD[index_T]
            codes_TQ[:, level] = index_T
        return codes_TQ

    def generate_codes(
        self,
        hidden_TD,
        mog_head,
        *,
        num_iter: int,
        exponent: float,
        top_p: float | None = None,
        noise_scale: float = 1.0,
        guidance_scale: float = 0.0,
    ):
        if guidance_scale > 0:
            hidden_TD, uncond_TD = hidden_TD.chunk(2)
        frames = hidden_TD.shape[0]
        codes_TQ = torch.zeros(
            frames, self.num_quantizers, dtype=torch.long, device=hidden_TD.device
        )
        rates = torch.linspace(0.0, 1.0, num_iter + 1, device=hidden_TD.device)[:-1]
        masking = (1.0 - rates.pow(exponent)).pow(1.0 / exponent)
        counts = torch.ceil(masking * self.num_quantizers).long()
        counts = counts - torch.cat([counts[1:], counts.new_zeros(1)])

        assigned = 0
        for count in counts.tolist():
            if count == 0:
                continue
            depth_TD = self.embed_code(self._depth_sum(codes_TQ, assigned))
            fed_TD = depth_TD + hidden_TD
            if guidance_scale > 0:
                fed_TD = torch.cat([fed_TD, depth_TD + uncond_TD])
            mean_TD, log_std_T1 = mog_head.infer(
                fed_TD, guidance_scale=guidance_scale, top_p=top_p
            )
            sampled_TD = (
                mean_TD
                + torch.exp(log_std_T1) * torch.randn_like(mean_TD) * noise_scale
            )
            codes_TQ = self.quantise(sampled_TD, codes_TQ, assigned, count)
            assigned += count
        return codes_TQ

    def _depth_sum(self, codes_TQ, levels: int):
        if levels == 0:
            return torch.zeros(
                codes_TQ.shape[0],
                self.rvq_embs.shape[-1],
                device=codes_TQ.device,
                dtype=self.embed_code.weight.dtype,
            )
        padded_QCD = functional.pad(self.rvq_embs, (0, 0, 0, 1))
        return torch.stack([padded_QCD[q][codes_TQ[:, q]] for q in range(levels)]).sum(
            0
        )


TALKER_ARCH = "NemotronVoiceChatTalker"
TALKER_PREFIX = "tts_model.tts_model."


class NemotronVoiceChatTalker(nn.Module):
    def __init__(self, *, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.config = config
        speech = config.nemotron_speech
        tts_config = speech["tts_config"]
        codec_config = speech["codec_config"]
        self.llm = Gemma3ForCausalLM(config=config, quant_config=quant_config)
        talker_config = dict(
            hidden_size=tts_config["backbone_config"]["hidden_size"],
            vocab_size=speech["text_vocab_size"],
            char_vocab_size=speech["char_vocab_size"],
            num_quantizers=codec_config["num_quantizers"],
            codebook_size=codec_config["codebook_size"],
            latent_size=codec_config["latent_size"],
            char_encoder_config=tts_config["cas_config"]["backbone_config"],
        )
        self.talker = EarTtsTalker(talker_config)
        self.mog_head = MoGHead(
            dict(
                hidden_size=talker_config["hidden_size"],
                intermediate_size=tts_config["backbone_config"]["intermediate_size"],
                num_layers=3,
                num_predictions=talker_config["codebook_size"],
                out_size=talker_config["latent_size"],
                low_rank=64,
            )
        )
        self.register_buffer(
            "codec_silence_tokens",
            torch.zeros(talker_config["num_quantizers"], dtype=torch.long),
        )
        self.register_buffer(
            "audio_prompt_latent",
            torch.zeros(speech["prompt_frames"], talker_config["hidden_size"]),
        )
        hidden_size = talker_config["hidden_size"]
        max_batch = get_global_server_args().max_running_requests
        embed_dtype = torch.get_default_dtype()
        device = "cuda"
        self._fusion_buffer = torch.zeros(
            max_batch, hidden_size, dtype=embed_dtype, device=device
        )
        self._fusion_mask = torch.zeros(max_batch, dtype=torch.bool, device=device)
        self._hidden_out = torch.zeros(
            max_batch, hidden_size, dtype=embed_dtype, device=device
        )

    def get_attention_sliding_window_size(self):
        return self.llm.get_attention_sliding_window_size()

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **_):
        if input_embeds is None:
            batch = input_ids.shape[0]
            assert bool(
                self._fusion_mask[:batch].all()
            ), "talker decode step reached the model without fused inputs"
            input_embeds = self._fusion_buffer[:batch]
            self._fusion_mask[:batch] = False
        hidden = self.llm.model(input_ids, positions, forward_batch, input_embeds)
        if forward_batch.forward_mode.is_decode():
            self._hidden_out[: hidden.shape[0]] = hidden
        else:
            last_rows = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            self._hidden_out[: last_rows.shape[0]] = hidden[last_rows]
        return self.llm.logits_processor(
            input_ids, hidden, self.llm.model.embed_tokens, forward_batch
        )

    def load_weights(self, weights):
        speaker = self.config.nemotron_speech["speaker"]
        prompt_key = f"tts_model.audio_prompt_latents.{speaker}"
        backbone_weights = []
        talker_state = {}
        mog_state = {}
        for name, tensor in weights:
            if name == "tts_model.codec_silence_tokens":
                self.codec_silence_tokens.copy_(tensor)
            elif name == prompt_key:
                self.audio_prompt_latent.copy_(
                    tensor[0].to(self.audio_prompt_latent.dtype)
                )
            elif name.startswith(TALKER_PREFIX):
                local = name[len(TALKER_PREFIX) :]
                if local.startswith("backbone."):
                    backbone_weights.append(
                        ("model." + local[len("backbone.") :], tensor)
                    )
                elif local.startswith("mog_head."):
                    mog_state[local[len("mog_head.") :]] = tensor
                else:
                    talker_state[local] = tensor
        self.llm.load_weights(backbone_weights)
        self.talker.load_state_dict(talker_state, strict=True)
        self.mog_head.load_state_dict(mog_state, strict=True)
        # float32 outside the backbone: the head's sample is quantised to its
        # nearest RVQ entry, and bfloat16 rounding picks neighbouring codes.
        self.talker.float()
        self.mog_head.float()
        self.audio_prompt_latent = self.audio_prompt_latent.float()
