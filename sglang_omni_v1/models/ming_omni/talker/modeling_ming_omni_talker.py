# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni talker model.

The internal LLM backbone (dense Qwen2, hidden=896), with CUDA graph
infrastructure, CFM/DiT/Aggregator modules and generation.
"""

from __future__ import annotations

import logging
import math
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from transformers import Qwen2Config, Qwen2Model, StaticCache

from .configuration_bailing_talker import MingOmniTalkerConfig
from .front.number_en import normalize_numbers
from .front.text_segment_cut import cut_text_by_semantic_length, is_chinese
from .front.toolkit import tokenize_mixed_text_iterator
from .talker_module.aggregator import Aggregator
from .talker_module.cfm import CFM, get_epss_timesteps
from .talker_module.dit import DiT

logger = logging.getLogger(__name__)

# ---------- Optional: onnxruntime for speaker embedding ----------
try:
    import onnxruntime

    _HAS_ONNX = True
except ImportError:
    onnxruntime = None  # type: ignore[assignment]
    _HAS_ONNX = False


class _IdentityNormalizer:
    """Fallback when TalkerTN (pynini) is not available."""

    def normalize(self, text: str) -> str:
        return text


class SpkembExtractor:
    """Extract speaker embeddings using CampPlus ONNX model."""

    def __init__(self, campplus_model: str, target_sr: int = 16000):
        if not _HAS_ONNX:
            raise ImportError("onnxruntime is required for SpkembExtractor")
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 2
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.target_sr = target_sr

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        return torch.tensor([embedding])

    def __call__(self, waveform, **kwargs) -> Optional[torch.Tensor]:
        return self._extract_spk_embedding(waveform)


class CFMGraphExecutor:
    def __init__(self, config, cfm, aggregator, stop_head):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.initialized = False
        self.last_hidden_state_placeholder = None
        self.his_lat_placeholder = None
        self.randn_like_placeholder = None
        self.t_placeholder = None
        self.sde_args_placeholder = None
        self.sde_rnd_placeholder = None
        self.gen_lat_placeholder = None
        self.inputs_embeds_placeholder = None
        self.stop_out_placeholder = None
        self.graph = None

    def execute(
        self, input_tensor, his_lat, cfg_strength=2.0, sigma=0.25, temperature=0.0
    ):
        bat_size, his_patch_size, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.config.patch_size, z_dim),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        t = get_epss_timesteps(
            self.config.steps, device=input_tensor.device, dtype=input_tensor.dtype
        )
        sde_rnd = torch.randn(
            (self.config.steps, *randn_tensor.shape),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

        if not self.initialized:
            self._initialize_graph(input_tensor, his_lat, randn_tensor, sde_rnd)

        self.last_hidden_state_placeholder.copy_(input_tensor)
        self.his_lat_placeholder.copy_(his_lat)
        self.randn_like_placeholder.copy_(randn_tensor)
        self.t_placeholder.copy_(t)
        self.sde_args_placeholder[0] = cfg_strength
        self.sde_args_placeholder[1] = sigma
        self.sde_args_placeholder[2] = temperature
        self.sde_rnd_placeholder.copy_(sde_rnd)

        self.graph.replay()

        gen_lat = torch.empty_like(self.gen_lat_placeholder)
        gen_lat.copy_(self.gen_lat_placeholder)
        inputs_embeds = torch.empty_like(self.inputs_embeds_placeholder)
        inputs_embeds.copy_(self.inputs_embeds_placeholder)
        stop_out = torch.empty_like(self.stop_out_placeholder)
        stop_out.copy_(self.stop_out_placeholder)

        return gen_lat, inputs_embeds, stop_out

    def _initialize_graph(self, input_tensor, his_lat, randn_tensor, sde_rnd):
        self.last_hidden_state_placeholder = torch.empty_like(input_tensor)
        self.his_lat_placeholder = torch.empty_like(his_lat)
        self.randn_like_placeholder = torch.empty_like(randn_tensor)
        self.t_placeholder = get_epss_timesteps(
            self.config.steps,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        self.sde_args_placeholder = torch.empty(
            3, device=input_tensor.device, dtype=input_tensor.dtype
        )
        self.sde_rnd_placeholder = torch.empty_like(sde_rnd)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.gen_lat_placeholder = self.cfm.sample(
                self.last_hidden_state_placeholder,
                self.his_lat_placeholder,
                self.randn_like_placeholder,
                self.t_placeholder,
                self.sde_args_placeholder,
                self.sde_rnd_placeholder,
            )
            self.inputs_embeds_placeholder = self.aggregator(self.gen_lat_placeholder)
            self.stop_out_placeholder = self.stop_head(
                self.last_hidden_state_placeholder[:, -1, :]
            ).softmax(dim=-1)

        self.initialized = True


class CFMGraphExecutorPool:
    def __init__(self, config, cfm, aggregator, stop_head, pool_size=5):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.pool_size = pool_size
        self.pool: Queue = Queue(maxsize=pool_size)
        self.lock = Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.pool_size):
            self.pool.put(
                CFMGraphExecutor(self.config, self.cfm, self.aggregator, self.stop_head)
            )

    def acquire(self):
        return self.pool.get()

    def release(self, executor):
        if isinstance(executor, CFMGraphExecutor):
            self.pool.put(executor)

    def execute(
        self, input_tensor, his_lat, cfg_strength=2.0, sigma=0.25, temperature=0.0
    ):
        executor = self.acquire()
        try:
            return executor.execute(
                input_tensor, his_lat, cfg_strength, sigma, temperature
            )
        finally:
            self.release(executor)


class MingOmniTalker(nn.Module):
    """Ming-Omni talker model.

    Submodules:
    - model: Qwen2Model (dense LLM backbone, hidden=896)
    - cfm: CFM(DiT) (flow matching, 28-layer DiT, hidden=1024)
    - aggregator: Aggregator (28-layer transformer, hidden=1152)
    - stop_head: nn.Linear(896, 2)
    - spk_head: nn.Linear(192, 896)
    """

    def __init__(self, config: MingOmniTalkerConfig):
        super().__init__()
        self.config = config

        # Qwen2 LLM backbone
        self.model_config = Qwen2Config(**config.llm_config)
        self.model = Qwen2Model(self.model_config)
        self.model.config._attn_implementation = "sdpa"

        self.latent_dim = config.latent_dim
        self.cfm = CFM(
            DiT(llm_cond_dim=self.model.config.hidden_size, **config.flowmodel),
            steps=config.steps,
        )
        self.aggregator = Aggregator(
            llm_input_dim=self.model.config.hidden_size,
            **config.aggregator,
        )

        self.stop_head = nn.Linear(self.model.config.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(
            config.spk_dim, self.model.config.hidden_size, bias=True
        )

        self.patch_size = config.patch_size
        self.his_patch_size = config.history_patch_size

        # --- External dependencies (set via setters) ---
        self.tokenizer = None
        self.normalizer: Any = _IdentityNormalizer()
        self.spkemb_extractor = None
        self.voice_json_dict: dict = {}

        # --- Internal state ---
        self.lock = threading.Lock()
        self.tts_speech_token_dict: dict = {}
        self.llm_end_dict: dict = {}
        self.vae_cache: dict = {}
        self.sil_holder_cache: dict = {}

        self.initialized = None
        self.initial_lock = threading.Lock()
        self.registered_prompt: dict = {}
        self.max_conc = config.max_conc
        self.executor = ThreadPoolExecutor(max_workers=self.max_conc)
        self.sampler_pool = CFMGraphExecutorPool(
            self.config,
            self.cfm,
            self.aggregator,
            self.stop_head,
            self.max_conc,
        )
        self.model_graph_pool: queue.Queue = queue.Queue()
        self.past_key_values = None
        for _ in range(self.max_conc):
            self.model_graph_pool.put((None, None, None, None, None, None, None))

    # ---- External dependency setters ----

    def set_tokenizer(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer

    def set_voice_presets(self, voice_dict: dict) -> None:
        self.voice_json_dict = voice_dict

    def set_spkemb_extractor(self, extractor) -> None:
        self.spkemb_extractor = extractor

    # ---- Weight loading ----

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Stream weights into model parameters.

        Weight mapping (checkpoint -> model):
        - model.* -> self.model.* (Qwen2 backbone, direct match)
        - cfm.model.* -> self.cfm.model.* (DiT, direct match)
        - aggregator.* -> self.aggregator.* (Aggregator, direct match)
        - stop_head.* -> self.stop_head.* (direct match)
        - spk_head.* -> self.spk_head.* (direct match)

        No weight name remapping needed — checkpoint names match nn.Module names.
        """
        params_dict = dict(self.named_parameters())
        loaded = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                logger.warning("Unexpected weight: %s", name)
                continue
            param = params_dict[name]
            if param.numel() == 1 and loaded_weight.numel() == 1:
                param.data.fill_(loaded_weight.item())
            else:
                assert (
                    param.size() == loaded_weight.size()
                ), f"Shape mismatch for {name}: param={param.size()}, weight={loaded_weight.size()}"
                param.data.copy_(loaded_weight)
            loaded.add(name)

        missing = set(params_dict.keys()) - loaded
        if missing:
            logger.warning(
                "Missing weights (%d): %s", len(missing), sorted(missing)[:20]
            )

    # ---- Forward (not used directly) ----

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    # ---- CUDA graph initialization ----

    def initial_graph(self, tokenizer=None):
        """Initialize CUDA graphs for generation.

        Args:
            tokenizer: If provided, sets the model tokenizer before graph init.
                       This is needed because the tokenizer must be available
                       for ``omni_audio_generation_func``.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer

        with self.initial_lock:
            if not self.initialized:
                for _ in range(self.max_conc):
                    this_uuid = str(uuid.uuid1())
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = []
                        self.llm_end_dict[this_uuid] = False
                        self.vae_cache[this_uuid] = {
                            "past_key_values": None,
                            "stream_state": (None, None, None),
                        }
                        self.sil_holder_cache[this_uuid] = None

                    prompt = (
                        "Please generate speech based on the following description.\n"
                    )
                    text = "Initialize compilation graph"
                    future = self.executor.submit(
                        self.llm_job,
                        prompt,
                        text,
                        None,
                        None,
                        "",
                        None,
                        None,
                        this_uuid,
                    )
                    future.result()

                    with self.lock:
                        self.tts_speech_token_dict.pop(this_uuid)
                        self.llm_end_dict.pop(this_uuid)
                        self.vae_cache.pop(this_uuid)
                        self.sil_holder_cache.pop(this_uuid)

                self.initialized = True

    # ---- Generation ----

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        prompt_wav_lat=None,
        min_new_token=3,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
    ):
        step = 0
        target_dtype = torch.bfloat16
        inputs_embeds = inputs_embeds.to(dtype=target_dtype)

        his_lat = torch.zeros(
            1,
            self.his_patch_size,
            self.latent_dim,
            device=self.device,
            dtype=target_dtype,
        )
        if prompt_wav_lat is not None:
            prompt_wav_lat = prompt_wav_lat.to(dtype=target_dtype)
            start_index = self.his_patch_size - prompt_wav_lat.size(1)
            if start_index < 0:
                his_lat[:] = prompt_wav_lat[:, -start_index:, :]
            else:
                his_lat[:, start_index:, :] = prompt_wav_lat

        max_cache_len = 512

        (
            past_key_values,
            inputs_embeds_placeholder,
            cache_position_placeholder,
            position_ids_placeholder,
            attention_mask_placeholder,
            outputs_placeholder,
            model_graph,
        ) = self.model_graph_pool.get()

        if past_key_values is None:
            past_key_values = StaticCache(
                config=self.model.config,
                max_batch_size=1,
                max_cache_len=max_cache_len,
                device=self.model.device,
                dtype=target_dtype,
            )
        else:
            if hasattr(past_key_values, "reset"):
                past_key_values.reset()
            elif hasattr(past_key_values, "key_cache"):
                for layer_idx in range(len(past_key_values.key_cache)):
                    past_key_values.key_cache[layer_idx].zero_()
                    past_key_values.value_cache[layer_idx].zero_()
            else:
                for layer in past_key_values.layers:
                    layer.keys.zero_()
                    layer.values.zero_()

        prefill_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(input_ids.shape).to(input_ids.device)
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(
            (attention_mask == 0), 1
        )

        max_decode_steps = (max_cache_len - prefill_len) // self.patch_size

        while step < 1000 and step < max_decode_steps:
            if step == 0:
                prefill_cache_position = torch.arange(
                    0, prefill_len, device=inputs_embeds.device
                )
                outputs = self.model(
                    position_ids=position_ids,
                    cache_position=prefill_cache_position,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    output_hidden_states=True,
                )
            else:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

                if model_graph is None:
                    model_graph = torch.cuda.CUDAGraph()
                    inputs_embeds_placeholder = torch.empty_like(inputs_embeds)
                    position_ids_placeholder = None
                    attention_mask_placeholder = None
                    cache_position_placeholder = torch.empty_like(cache_position)

                    inputs_embeds_placeholder.copy_(inputs_embeds)
                    cache_position_placeholder.copy_(cache_position)

                    with torch.cuda.graph(model_graph):
                        outputs_placeholder = self.model(
                            position_ids=position_ids_placeholder,
                            cache_position=cache_position_placeholder,
                            attention_mask=attention_mask_placeholder,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds_placeholder,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                else:
                    inputs_embeds_placeholder.copy_(inputs_embeds)
                    if cache_position_placeholder is not None:
                        cache_position_placeholder.copy_(cache_position)
                    model_graph.replay()

                outputs = outputs_placeholder

            hidden_out = outputs.hidden_states[-1][:, -1:, :]

            gen_lat, inputs_embeds, stop_out = self.sampler_pool.execute(
                hidden_out,
                his_lat,
                cfg,
                sigma,
                temperature,
            )

            if self.his_patch_size == self.patch_size:
                his_lat = gen_lat
            elif self.his_patch_size > self.patch_size:
                his_lat = torch.cat(
                    [his_lat[:, self.patch_size - self.his_patch_size :], gen_lat],
                    dim=1,
                )
            else:
                raise NotImplementedError

            if step > min_new_token and stop_out.cpu()[0, 1] > 0.5:
                yield gen_lat, True
                break

            yield gen_lat, False
            step += 1

        self.model_graph_pool.put(
            (
                past_key_values,
                inputs_embeds_placeholder,
                cache_position_placeholder,
                position_ids_placeholder,
                attention_mask_placeholder,
                outputs_placeholder,
                model_graph,
            )
        )

    def omni_audio_generation_func(
        self,
        prompt,
        text,
        spk_emb,
        instruction,
        prompt_text=None,
        prompt_wav_lat=None,
        prompt_wav_emb=None,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
    ):
        assert (
            self.tokenizer is not None
        ), "Tokenizer not set. Call set_tokenizer() first."
        tokenizer = self.tokenizer

        spk_emb_prompt: list = []
        if spk_emb is not None:
            for i, se in enumerate(spk_emb):
                spk_emb_prompt.extend(
                    tokenizer.encode(f"  speaker_{i+1}:")
                    + tokenizer.encode("<|vision_start|>")
                    + tokenizer.encode("<|vision_pad|>")
                    + tokenizer.encode("<|vision_end|>\n")
                )

        instruction_prompt: list = []
        if instruction is not None:
            instruction_prompt = tokenizer.encode(instruction) + tokenizer.encode(
                "<|im_end|>"
            )

        prompt_text_token: list = []
        prompt_latent_token: list = []
        if prompt_wav_emb is not None and prompt_text is not None:
            prompt_text_token = tokenizer.encode(prompt_text)
            prompt_latent_token = tokenizer.encode(
                "<audioPatch>"
            ) * prompt_wav_emb.size(1)

        prompt2 = tokenizer.encode(" Text input:\n")
        if (
            "Genre: " in text
            and "Mood: " in text
            and "Instrument: " in text
            and "Theme: " in text
            and "Duration: " in text
        ):
            prompt2 = []

        input_part = (
            tokenizer.encode(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            )
            + tokenizer.encode("<|im_start|>user\n")
            + tokenizer.encode(prompt)
            + spk_emb_prompt
            + prompt2
            + prompt_text_token
            + tokenizer.encode(text)
            + tokenizer.encode("<|im_end|>\n")
            + tokenizer.encode("<|im_start|>assistant\n")
            + instruction_prompt
            + tokenizer.encode("<audio>")
            + prompt_latent_token
        )

        logger.info("Talker input: %r", tokenizer.decode(input_part)[:200])

        input_ids = (
            torch.tensor(input_part, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        inputs_embeds = self.model.get_input_embeddings()(input_ids).to(
            device=self.device,
            dtype=torch.bfloat16,
        )

        if spk_emb is not None:
            spk_token_id = tokenizer.encode("<|vision_start|>")
            assert len(spk_token_id) == 1
            spk_indices = torch.where(input_ids[0] == spk_token_id[0])[0]
            assert len(spk_indices) > 0
            for i, se in enumerate(spk_emb):
                inputs_embeds[0, spk_indices[i] + 1] = se.to(dtype=torch.bfloat16)

        if prompt_wav_emb is not None and prompt_text is not None:
            audio_token_id = tokenizer.encode("<audio>")
            assert len(audio_token_id) == 1
            audio_indices = torch.where(input_ids[0] == audio_token_id[0])[0]
            assert len(audio_indices) > 0
            inputs_embeds[
                0,
                audio_indices[0] + 1 : audio_indices[0] + 1 + prompt_wav_emb.size(1),
                :,
            ] = prompt_wav_emb[0].to(dtype=torch.bfloat16)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for audio_token in self.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                prompt_wav_lat=prompt_wav_lat,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
            ):
                yield audio_token

    def token2wav(
        self, audio_detokenizer, token, cache=None, stream=False, last_chunk=False
    ):
        speech, stream_state, past_key_values = audio_detokenizer.decode(
            torch.cat(token, dim=1),
            use_cache=stream,
            **cache,
            last_chunk=last_chunk,
        )
        new_cache = {"past_key_values": past_key_values, "stream_state": stream_state}
        return speech[0].detach().float(), new_cache

    @staticmethod
    def silence_holder(
        speech, sample_rate, sil_cache=None, last_chunk=True, sil_th=1e-3, last_sil=0.3
    ):
        if speech.numel() == 0:
            assert not last_chunk
            return speech, sil_cache

        frame_step, frame_size = int(sample_rate * 0.1), int(sample_rate * 0.1)
        if sil_cache is None:
            sil_cache = {"holder": [], "buffer": []}
        if sil_cache["buffer"]:
            speech = torch.cat([*sil_cache["buffer"], speech], dim=-1)
            sil_cache["buffer"] = []
        if speech.shape[-1] < frame_size:
            sil_cache["buffer"].append(speech)
            if last_chunk:
                speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
                return speech[..., : int(last_sil * sample_rate)], sil_cache
            return (
                torch.zeros(
                    (*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype
                ),
                sil_cache,
            )

        num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
        cur_len = (num_frame - 1) * frame_step + frame_size
        if speech.shape[-1] > cur_len:
            sil_cache["buffer"].append(speech[..., cur_len:])
            speech = speech[..., :cur_len]
        spe_frames = speech.unfold(-1, frame_size, frame_step)
        scores = spe_frames.abs().mean(dim=-1)
        scores = scores.mean(dim=list(range(scores.dim() - 1)))
        idx = scores.shape[0] - 1
        while idx >= 0:
            if scores[idx] > sil_th:
                break
            idx -= 1
        if idx < 0:
            sil_cache["holder"].append(speech)
            if last_chunk:
                speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
                return speech[..., : int(last_sil * sample_rate)], sil_cache
            return (
                torch.zeros(
                    (*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype
                ),
                sil_cache,
            )
        non_sil_len = idx * frame_step + frame_size
        if last_chunk:
            non_sil_len += int(last_sil * sample_rate)
        speech = torch.cat([*sil_cache["holder"], speech[..., :non_sil_len]], dim=-1)
        sil_cache["holder"] = []
        if non_sil_len < speech.shape[-1]:
            sil_cache["holder"].append(speech[..., non_sil_len:])
        return speech, sil_cache

    def llm_job(
        self,
        prompt,
        text,
        spk_emb,
        instruction,
        prompt_text,
        prompt_wav_lat,
        prompt_wav_emb,
        this_uuid,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
    ):
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            for audio_token in self.omni_audio_generation_func(
                prompt=prompt,
                text=text,
                spk_emb=spk_emb,
                instruction=instruction,
                prompt_text=prompt_text,
                prompt_wav_lat=prompt_wav_lat,
                prompt_wav_emb=prompt_wav_emb,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
            ):
                torch.cuda.current_stream().synchronize()
                self.tts_speech_token_dict[this_uuid].append(audio_token)
        self.llm_end_dict[this_uuid] = True

    def tts_job(
        self,
        prompt,
        text,
        spk_emb,
        instruction,
        audio_detokenizer,
        prompt_text,
        prompt_wav_lat,
        prompt_wav_emb,
        stream,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
    ):
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            this_uuid = str(uuid.uuid1())
            with self.lock:
                self.tts_speech_token_dict[this_uuid] = []
                self.llm_end_dict[this_uuid] = False
                self.vae_cache[this_uuid] = {
                    "past_key_values": None,
                    "stream_state": (None, None, None),
                }
                self.sil_holder_cache[this_uuid] = None

            future = self.executor.submit(
                self.llm_job,
                prompt,
                text,
                spk_emb,
                instruction,
                prompt_text,
                prompt_wav_lat,
                prompt_wav_emb,
                this_uuid,
                cfg,
                sigma,
                temperature,
            )

            if stream:
                token_offset = 0
                while True:
                    if future.done():
                        exc = future.exception()
                        if exc:
                            raise exc
                    time.sleep(0.1)
                    nxt = len(self.tts_speech_token_dict[this_uuid])
                    if nxt > token_offset:
                        this_tts_speech_token = self.tts_speech_token_dict[this_uuid][
                            token_offset:nxt
                        ]
                        last_chunk = this_tts_speech_token[-1][-1]
                        this_tts_speech_token = [ii[0] for ii in this_tts_speech_token]
                        this_tts_speech, self.vae_cache[this_uuid] = self.token2wav(
                            audio_detokenizer=audio_detokenizer,
                            token=this_tts_speech_token,
                            cache=self.vae_cache[this_uuid],
                            stream=True,
                            last_chunk=last_chunk,
                        )
                        token_offset = nxt
                        yield {"tts_speech": this_tts_speech.cpu()}

                    if self.llm_end_dict[this_uuid] and token_offset == len(
                        self.tts_speech_token_dict[this_uuid]
                    ):
                        break
                future.result()
            else:
                future.result()
                this_tts_speech_token = self.tts_speech_token_dict[this_uuid]
                this_tts_speech_token = [ii[0] for ii in this_tts_speech_token]
                this_tts_speech, self.vae_cache[this_uuid] = self.token2wav(
                    audio_detokenizer=audio_detokenizer,
                    token=this_tts_speech_token,
                    cache=self.vae_cache[this_uuid],
                    stream=False,
                    last_chunk=True,
                )
                this_tts_speech, self.sil_holder_cache[this_uuid] = self.silence_holder(
                    this_tts_speech,
                    audio_detokenizer.config.sample_rate,
                    self.sil_holder_cache[this_uuid],
                    True,
                )
                yield {"tts_speech": this_tts_speech.cpu()}

            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()

            with self.lock:
                self.tts_speech_token_dict.pop(this_uuid)
                self.llm_end_dict.pop(this_uuid)
                self.vae_cache.pop(this_uuid)
                self.sil_holder_cache.pop(this_uuid)

    def register_prompt_wav(self, prompt_wav_path, audio_detokenizer):
        if isinstance(prompt_wav_path, str):
            prompt_wav_path = [prompt_wav_path]

        speech_parts = []
        spk_emb_list = []
        for x in prompt_wav_path:
            speech_tmp, sample_rate = torchaudio.load(x, backend="soundfile")
            speech_tmp1 = speech_tmp.clone()
            if sample_rate != audio_detokenizer.config.sample_rate:
                speech_tmp = torchaudio.transforms.Resample(
                    sample_rate, audio_detokenizer.config.sample_rate
                )(speech_tmp)
            speech_parts.append(speech_tmp)

            if self.spkemb_extractor is not None:
                if sample_rate != 16000:
                    speech_tmp1 = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=16000
                    )(speech_tmp1)
                se = self.spkemb_extractor(speech_tmp1)
                se = self.spk_head(se.to(device=self.device, dtype=self.dtype))
                spk_emb_list.append(se)

        speech = torch.cat(speech_parts, dim=-1)

        patch_pt = (
            audio_detokenizer.encoder.hop_size
            * max(1, audio_detokenizer.encoder.patch_size)
            * self.patch_size
        )
        if speech.shape[-1] % patch_pt != 0:
            pad_len = (speech.shape[1] + patch_pt - 1) // patch_pt * patch_pt
            pad_speech = torch.zeros(
                (speech.shape[0], pad_len), dtype=speech.dtype, device=speech.device
            )
            pad_speech[:, -speech.shape[1] :] = speech
            speech = pad_speech
        prompt_wav_lat, _ = audio_detokenizer.encode_latent(
            speech.to(dtype=torch.bfloat16, device=self.device),
            torch.tensor([speech.size(1)], dtype=torch.long, device=self.device),
        )
        assert prompt_wav_lat.shape[1] % self.patch_size == 0
        prompt_wav_lat = prompt_wav_lat.reshape(
            -1, self.patch_size, prompt_wav_lat.shape[-1]
        )
        prompt_wav_emb = self.aggregator(prompt_wav_lat)
        prompt_wav_lat = prompt_wav_lat.reshape(1, -1, prompt_wav_lat.shape[-1])
        prompt_wav_emb = prompt_wav_emb.reshape(1, -1, prompt_wav_emb.shape[-1])

        key = (
            "|".join(prompt_wav_path)
            if len(prompt_wav_path) > 1
            else prompt_wav_path[0]
        )
        self.registered_prompt[key] = {
            "prompt_wav_lat": prompt_wav_lat,
            "prompt_wav_emb": prompt_wav_emb,
            "spk_emb": spk_emb_list if spk_emb_list else None,
        }
        logger.info("register_prompt_wav: %s", key)

    def get_prompt_emb(
        self,
        prompt_wav_path,
        audio_detokenizer,
        use_spk_emb=False,
        use_zero_spk_emb=False,
    ):
        if prompt_wav_path is None:
            if not use_zero_spk_emb:
                return None, None, None
            return (
                None,
                None,
                torch.zeros(
                    1,
                    self.model.config.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
        if isinstance(prompt_wav_path, list):
            key = "|".join(prompt_wav_path)
        else:
            key = prompt_wav_path
        if key not in self.registered_prompt:
            self.register_prompt_wav(prompt_wav_path, audio_detokenizer)
        msg = self.registered_prompt[key]
        spk_emb = msg["spk_emb"] if use_spk_emb else None
        return msg["prompt_wav_lat"], msg["prompt_wav_emb"], spk_emb

    def _run_tts_segments(
        self,
        text,
        prompt,
        instruction,
        spk_emb,
        audio_detokenizer,
        prompt_text,
        prompt_wav_lat,
        prompt_wav_emb,
        stream,
        max_length=50,
    ):
        count = 0
        cache_position: dict = {}
        wds_lg_zh = 6.07
        wds_lg_en = 16
        streaming_text: list = []

        tts_text_list = tokenize_mixed_text_iterator(text)

        for i, ele in enumerate(tts_text_list):
            if len(ele) == 0:
                continue

            should_process = False
            if ele[-1] in "\uff01\uff1f\u3002\uff0c!?" and (
                len(streaming_text) >= 12 or (count > 0 and len(streaming_text) >= 8)
            ):
                should_process = True
                streaming_text.append(ele)
            elif (
                ele[-1] == "."
                and (
                    len(streaming_text) >= 12
                    or (count > 0 and len(streaming_text) >= 8)
                )
                and not bool(
                    re.search(
                        r"[0-9]", streaming_text[-1][-1] if streaming_text else ""
                    )
                )
            ):
                should_process = True
                streaming_text.append(ele)
            elif ele[-1] == "\n":
                if len(streaming_text) > 0:
                    if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                        if bool(re.search(r"[\u4e00-\u9fff]", streaming_text[-1][-1])):
                            ele = "\uff0c"
                            streaming_text.append(ele)
                    else:
                        if len(ele) > 1 and bool(re.search(r"[a-zA-Z]", ele[-2])):
                            ele = ele[:-1] + "."
                        else:
                            ele = ele[:-1]
                        streaming_text.append(ele)
                if len(streaming_text) >= 12 or (
                    count > 0 and len(streaming_text) >= 8
                ):
                    should_process = True
            else:
                streaming_text.append(ele)
                continue

            if should_process:
                yield from self._process_segment(
                    "".join(streaming_text),
                    prompt,
                    instruction,
                    spk_emb,
                    audio_detokenizer,
                    prompt_text,
                    prompt_wav_lat,
                    prompt_wav_emb,
                    stream,
                    count,
                    cache_position,
                    max_length,
                    wds_lg_zh,
                    wds_lg_en,
                )
                count += 1
                streaming_text = []

        if streaming_text and re.search(
            r"[a-zA-Z\u4e00-\u9fff1-9]", "".join(streaming_text)
        ):
            yield from self._process_segment(
                "".join(streaming_text),
                prompt,
                instruction,
                spk_emb,
                audio_detokenizer,
                prompt_text,
                prompt_wav_lat,
                prompt_wav_emb,
                stream,
                count,
                cache_position,
                max_length,
                wds_lg_zh,
                wds_lg_en,
            )

    def _process_segment(
        self,
        streaming_text,
        prompt,
        instruction,
        spk_emb,
        audio_detokenizer,
        prompt_text,
        prompt_wav_lat,
        prompt_wav_emb,
        stream,
        count,
        cache_position,
        max_length,
        wds_lg_zh,
        wds_lg_en,
    ):
        sub_output_dict = cut_text_by_semantic_length(streaming_text, max_length)
        text_list = sub_output_dict["fragments"]
        if not text_list:
            return

        for text_ori in text_list:
            length = len(text_ori)
            if len(cache_position) == 0:
                cache_position.update({count: (0, length - 1)})
            else:
                end_idx = list(cache_position.values())[-1][1] + 1
                cache_position.update({count: (end_idx, end_idx + length - 1)})

            if not is_chinese(text_ori):
                text = normalize_numbers(text_ori)
                wds_lg = wds_lg_en
            else:
                text = text_ori
                wds_lg = wds_lg_zh

            text = self.normalizer.normalize(text)
            if text and text[0] == "\uff0c":
                text = text[1:]

            use_stream = stream and (count == 0)
            all_wavs: list = []

            for idx, this_tts_speech_dict in enumerate(
                self.tts_job(
                    prompt=prompt,
                    text=text,
                    spk_emb=spk_emb,
                    instruction=instruction,
                    audio_detokenizer=audio_detokenizer,
                    prompt_text=prompt_text,
                    prompt_wav_lat=prompt_wav_lat,
                    prompt_wav_emb=prompt_wav_emb,
                    stream=use_stream,
                )
            ):
                tts_speech = this_tts_speech_dict["tts_speech"]
                if (
                    all_wavs
                    and torch.cat(all_wavs, dim=-1).shape[1]
                    / audio_detokenizer.config.sample_rate
                    * (16000 / 5818)
                    >= len(text)
                    and torch.cat(all_wavs, dim=-1).shape[1]
                    / audio_detokenizer.config.sample_rate
                    > 2
                ):
                    break

                this_dura = float(
                    tts_speech.shape[-1] / audio_detokenizer.config.sample_rate
                )
                if use_stream:
                    if idx == 0:
                        this_start_idx = 0
                        this_end_idx = min(math.ceil(this_dura * wds_lg), length) - 1
                    else:
                        this_start_idx = min(
                            list(cache_position.values())[-1][1] + 1, length - 1
                        )
                        this_end_idx = (
                            min(
                                (math.ceil(this_dura * wds_lg) + this_start_idx), length
                            )
                            - 1
                        )
                    cache_position.update(
                        {f"{count}_{idx}": (this_start_idx, this_end_idx)}
                    )
                    this_text_ori = (
                        ""
                        if this_start_idx == this_end_idx
                        else text_ori[this_start_idx : this_end_idx + 1]
                    )
                    all_wavs.append(tts_speech)
                    yield tts_speech, this_text_ori, cache_position[
                        f"{count}_{idx}"
                    ], this_dura * 1000
                else:
                    all_wavs.append(tts_speech)
                    yield tts_speech, text_ori, cache_position[count], this_dura * 1000

    def omni_audio_generation(
        self,
        tts_text,
        voice_name="DB30",
        prompt_text=None,
        prompt_wav_path=None,
        max_length=50,
        audio_detokenizer=None,
        stream=False,
        **kwargs,
    ):
        text = tts_text
        prompt = "Please generate speech based on the following description.\n"
        instruction = None

        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            self.initial_graph()

            if voice_name is not None and voice_name in self.voice_json_dict:
                assert prompt_wav_path is None and prompt_text is None
                prompt_text = self.voice_json_dict[voice_name]["prompt_text"]
                prompt_wav_path = self.voice_json_dict[voice_name]["prompt_wav_path"]

            prompt_wav_lat, prompt_wav_emb, spk_emb = self.get_prompt_emb(
                prompt_wav_path,
                audio_detokenizer,
                use_spk_emb=True,
                use_zero_spk_emb=False,
            )

            yield from self._run_tts_segments(
                text,
                prompt,
                instruction,
                spk_emb,
                audio_detokenizer,
                prompt_text,
                prompt_wav_lat,
                prompt_wav_emb,
                stream,
                max_length,
            )

    def instruct_audio_generation(
        self,
        prompt,
        text,
        use_spk_emb=False,
        use_zero_spk_emb=False,
        instruction=None,
        prompt_wav_path=None,
        prompt_text=None,
        max_decode_steps=200,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
        max_length=50,
        audio_detokenizer=None,
        stream=False,
        taskname="TTS",
        **kwargs,
    ):
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            self.initial_graph()

            prompt_wav_lat, prompt_wav_emb, spk_emb = self.get_prompt_emb(
                prompt_wav_path,
                audio_detokenizer,
                use_spk_emb=use_spk_emb,
                use_zero_spk_emb=use_zero_spk_emb,
            )

            if taskname in [
                "TTA",
                "BGM",
                "STYLE",
                "SPEECH_BGM",
                "SPEECH_SOUND",
                "PODCAST",
            ]:
                for this_tts_speech_dict in self.tts_job(
                    prompt=prompt,
                    text=text,
                    spk_emb=spk_emb,
                    instruction=instruction,
                    audio_detokenizer=audio_detokenizer,
                    prompt_text=prompt_text,
                    prompt_wav_lat=prompt_wav_lat,
                    prompt_wav_emb=prompt_wav_emb,
                    stream=stream,
                    cfg=cfg,
                    sigma=sigma,
                    temperature=temperature,
                ):
                    yield this_tts_speech_dict["tts_speech"], None, None, None
            else:
                yield from self._run_tts_segments(
                    text,
                    prompt,
                    instruction,
                    spk_emb,
                    audio_detokenizer,
                    prompt_text,
                    prompt_wav_lat,
                    prompt_wav_emb,
                    stream,
                    max_length,
                )
