"""Append-only video observations, reply generation, and layer-50 conditioning."""

from dataclasses import dataclass, field
from threading import Event
from typing import ClassVar, NotRequired, TypedDict

import numpy as np
import torch
from accelerate import dispatch_model
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.cache_utils import Cache

from sglang_omni.models.duo import intent_conditioning_compat as intent_conditioning


class IntentVisual(TypedDict):
    patches: torch.Tensor
    grid: torch.Tensor | list[int]
    merge: NotRequired[int]


@dataclass(kw_only=True)
class LiveIntentState:
    record: dict[str, object]
    visuals: dict[tuple[str, int], dict[str, object]] = field(default_factory=dict)
    bodies: dict[int, list[int]] = field(default_factory=dict)
    input_ids: torch.Tensor | None = None
    past_key_values: Cache | None = None
    shared_h50: torch.Tensor | None = None
    description_tokens: int = 0
    user_token_ids: list[int] = field(default_factory=list)
    input_keep: list[bool] = field(default_factory=list)
    position_ids: torch.Tensor | None = None
    next_position: int = 0


@dataclass(frozen=True, kw_only=True)
class LivePlan:
    shared_h50: torch.Tensor
    plan_h50: torch.Tensor
    body_ids: list[int]
    text: str
    prefix_tokens: int
    reused_tokens: int
    kv_tokens: int
    description_tokens: int = 0
    graph_replays: int = 0
    evicted_tokens: int = 0
    serialization_seconds: float = 0.0
    position_seconds: float = 0.0
    worker_seconds: float = 0.0
    prefill_submit_seconds: float = 0.0
    decode_wall_seconds: float = 0.0
    h50_return_seconds: float = 0.0


class LiveIntentPlanner:
    incremental_inputs: ClassVar[bool] = False

    def __init__(
        self,
        processor: Qwen3VLProcessor,
        model: PeftModel | Qwen3VLForConditionalGeneration,
        *,
        max_context_tokens: int,
        max_new_tokens: int,
        vision_device: str | None = None,
    ) -> None:
        self.processor = processor
        self.model = model
        self.base_model = (
            model.get_base_model() if isinstance(model, PeftModel) else model
        )
        if vision_device is not None:
            dispatch_model(
                self.base_model,
                device_map={
                    "model.visual": vision_device,
                    "model.language_model": str(self.device),
                    "lm_head": str(self.device),
                },
                main_device=self.device,
            )
        else:
            pass
        self.kv_window_tokens: int | None = None
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer = processor.tokenizer
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        allowed = {
            self.tokenizer.convert_tokens_to_ids(token)
            for token in intent_conditioning.intent.SPECIALS
            if token == "<wait>" or token.startswith("<emo_")
        } | {self.end_token_id}
        self.forbidden_token_ids = sorted(set(self.tokenizer.all_special_ids) - allowed)
        self.states: dict[str, LiveIntentState] = {}

    @property
    def device(self) -> torch.device:
        return self.base_model.model.language_model.embed_tokens.weight.device

    def visual_patch(self, frames: np.ndarray) -> IntentVisual:
        return intent_conditioning.visual_patch(self.processor, frames)

    def serialize(
        self, state: LiveIntentState, block_index: int, block_duration_ms: float
    ) -> dict[str, object]:
        return intent_conditioning.serialize(
            state.record,
            self.tokenizer,
            state.visuals,
            stop_block=block_index,
            assistant_bodies=state.bodies,
            generation=True,
            block_duration_ms=block_duration_ms,
        )

    def open(self, session_id: str, description: dict[str, str]) -> None:
        self.states[session_id] = LiveIntentState(
            record={
                "sample_id": session_id,
                "blocks": [],
                "assistant_description_override": description,
                "canonical": {"source": {"metadata": ""}, "text_tokens": []},
            }
        )

    def system_prompt_fragments(self) -> tuple[str, str]:
        return "", ""

    def prepare_prefix(
        self,
        session_id: str,
        frames: np.ndarray,
        user_text: str,
        block_index: int,
        frames_per_block: int,
        block_duration_ms: float,
        *,
        prepared_visual: IntentVisual | None = None,
    ) -> tuple[
        LiveIntentState, dict[str, object], dict[str, object], torch.Tensor, int
    ]:
        state = self.states[session_id]
        block = {
            "start_ms": block_index * block_duration_ms,
            "end_ms": (block_index + 1) * block_duration_ms,
            "frame_start": block_index * frames_per_block,
            "frame_stop": (block_index + 1) * frames_per_block,
        }
        state.record["blocks"].append(block)
        state.user_token_ids = self.tokenizer.encode(
            user_text, add_special_tokens=False
        )
        state.record["canonical"]["text_tokens"].extend(
            {
                "role": "user",
                "token_id": token_id,
                "start_ms": block["start_ms"],
                "commit_ms": (block["start_ms"] + block["end_ms"]) / 2,
            }
            for token_id in state.user_token_ids
        )
        visual = (
            self.visual_patch(frames) if prepared_visual is None else prepared_visual
        )
        state.visuals["user", block_index] = visual
        prefix = self.serialize(state, block_index, block_duration_ms)
        input_ids = prefix["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            pass
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"Intent serializer must return [1, tokens], got {input_ids.shape}"
            )
        else:
            pass
        reused_tokens = 0 if state.input_ids is None else state.input_ids.shape[1]
        if self.incremental_inputs and state.input_ids is not None:
            input_ids = torch.cat((state.input_ids, input_ids), dim=1)
            prefix["keep"] = state.input_keep + prefix["keep"]
        else:
            pass
        if (
            self.kv_window_tokens is None
            and input_ids.shape[1] + self.max_new_tokens > self.max_context_tokens
        ):
            raise ValueError("Intent context capacity reached; start a new session")
        elif (
            not self.incremental_inputs
            and state.input_ids is not None
            and not torch.equal(input_ids[:, :reused_tokens], state.input_ids)
        ):
            raise ValueError("Committed Intent input changed; start a new session")
        else:
            pass
        return state, visual, prefix, input_ids, reused_tokens

    @torch.inference_mode()
    def append(
        self,
        session_id: str,
        frames: np.ndarray,
        user_text: str,
        block_index: int,
        cancelled: Event,
        *,
        frames_per_block: int = 4,
        block_duration_ms: float = 1000 / 6,
        prepared_visual: IntentVisual | None = None,
    ) -> LivePlan:
        state, visual, prefix, input_ids, reused_tokens = self.prepare_prefix(
            session_id,
            frames,
            user_text,
            block_index,
            frames_per_block,
            block_duration_ms,
            prepared_visual=prepared_visual,
        )
        modalities = (input_ids == self.model.config.video_token_id).long() * 2
        grid = prefix["video_grid_thw"]
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.long)
        else:
            pass
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        else:
            pass
        positions, _ = self.base_model.model.get_rope_index(
            input_ids,
            modalities,
            video_grid_thw=grid,
            attention_mask=torch.ones_like(input_ids),
        )
        captured: list[torch.Tensor] = []

        def capture(
            module: torch.nn.Module,
            arguments: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden.detach())

        hook = self.base_model.model.language_model.layers[49].register_forward_hook(
            capture
        )
        device = self.device
        body_ids: list[int] = []
        plan_hidden: list[torch.Tensor] = []
        generated_ids: list[int] = []
        try:
            output = self.model(
                input_ids=input_ids[:, reused_tokens:].to(device),
                attention_mask=torch.ones_like(input_ids, device=device),
                position_ids=positions[:, :, reused_tokens:].to(device),
                pixel_values_videos=visual["patches"].to(device),
                video_grid_thw=torch.as_tensor(visual["grid"], device=device).reshape(
                    -1, 3
                ),
                mm_token_type_ids=modalities[:, reused_tokens:].to(device),
                past_key_values=state.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            selected = torch.tensor(prefix["keep"][reused_tokens:], device=device)
            if state.shared_h50 is None:
                state.shared_h50 = captured[-1][0, selected].cpu().contiguous()
            else:
                pass
            cache = output.past_key_values
            next_position = int(positions.max()) + 1
            for token_index in range(self.max_new_tokens):
                if cancelled.is_set():
                    raise RuntimeError("Intent session cancelled")
                else:
                    pass
                logits = output.logits[:, -1].float()
                logits[:, self.forbidden_token_ids] = -torch.inf
                token_id = int(logits.argmax(dim=-1))
                generated_ids.append(token_id)
                output = self.model(
                    input_ids=torch.tensor([[token_id]], device=device),
                    attention_mask=torch.ones(
                        (1, input_ids.shape[1] + len(generated_ids)),
                        dtype=torch.long,
                        device=device,
                    ),
                    position_ids=torch.full(
                        (3, 1, 1),
                        next_position + token_index,
                        dtype=torch.long,
                        device=device,
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
                if token_id == self.end_token_id:
                    break
                else:
                    body_ids.append(token_id)
                    plan_hidden.append(captured[-1][0])
            else:
                raise RuntimeError("Intent reply did not finish within max_new_tokens")
            if not body_ids:
                raise RuntimeError("Intent emitted an empty plan")
            else:
                pass
        except BaseException:
            self.close(session_id)
            raise
        finally:
            hook.remove()
        state.bodies[block_index] = body_ids
        state.input_ids = torch.cat((input_ids, torch.tensor([generated_ids])), dim=1)
        state.past_key_values = cache
        return LivePlan(
            shared_h50=state.shared_h50,
            plan_h50=torch.cat(plan_hidden).cpu().contiguous(),
            body_ids=body_ids,
            text=self.tokenizer.decode(body_ids, skip_special_tokens=False),
            prefix_tokens=input_ids.shape[1],
            reused_tokens=reused_tokens,
            kv_tokens=cache.get_seq_length(),
            description_tokens=state.description_tokens,
        )

    def usage(self, session_id: str) -> tuple[int, int]:
        state = self.states.get(session_id)
        tokens = (
            0 if state is None or state.input_ids is None else state.input_ids.shape[1]
        )
        nbytes = 0
        if state is not None and state.past_key_values is not None:
            for layer in state.past_key_values.layers:
                nbytes += layer.keys.numel() * layer.keys.element_size()
                nbytes += layer.values.numel() * layer.values.element_size()
        else:
            pass
        return tokens, nbytes

    def close(self, session_id: str) -> None:
        self.states.pop(session_id, None)
