# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MiniCPM-V/MiniCPM-o pipeline components.

Covers:
- PipelineState serialization round-trip (io.py)
- build_llm_inputs / merge_for_llm / _prune_preprocessing_for_llm (merge.py)
- decode_events streaming / final decoding (merge.py)
- preprocessing_next / llm_next / vocoder_next routing (next_stage.py)
- build_minicpm_image_mm_inputs (preprocessor.py)
- MiniCPMVPipelineConfig / MiniCPMOPipelineConfig structure (config.py)

All tests run without a GPU and without loading real model weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# io.py
# ---------------------------------------------------------------------------


class TestPipelineState:
    """PipelineState serialization and field coverage."""

    def test_round_trip_empty(self):
        from sglang_omni.models.minicpm_v.io import PipelineState

        s = PipelineState()
        assert PipelineState.from_dict(s.to_dict()) == s

    def test_round_trip_with_prompt(self):
        from sglang_omni.models.minicpm_v.io import PipelineState

        s = PipelineState(
            prompt={"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "prompt_text": "hi"},
            mm_inputs={"image": {"tgt_sizes": [[16, 16]]}, "audio": {}},
            stream_state={"token_ids": [], "text": ""},
        )
        restored = PipelineState.from_dict(s.to_dict())
        assert restored.prompt == s.prompt
        assert restored.mm_inputs == s.mm_inputs

    def test_round_trip_with_vocoder_out(self):
        from sglang_omni.models.minicpm_v.io import PipelineState

        s = PipelineState(
            vocoder_out={"audio_data": [0.1, 0.2], "sample_rate": 22050},
        )
        restored = PipelineState.from_dict(s.to_dict())
        assert restored.vocoder_out == s.vocoder_out

    def test_from_dict_non_dict_input(self):
        from sglang_omni.models.minicpm_v.io import PipelineState

        # Should not raise
        s = PipelineState.from_dict(None)
        assert s.mm_inputs == {}
        assert s.encoder_inputs == {}

    def test_vocoder_out_not_persisted_when_empty(self):
        from sglang_omni.models.minicpm_v.io import PipelineState

        s = PipelineState()
        assert "vocoder_out" not in s.to_dict()

    def test_encoder_outs_round_trip(self):
        from sglang_omni.models.minicpm_v.io import PipelineState

        s = PipelineState(
            encoder_outs={
                "image_encoder": {"image_embeds": [[1.0, 2.0]], "tgt_sizes": [[8, 8]]},
                "audio_encoder": {"audio_embeds": [[0.5]], "audio_output_lengths": [16]},
            }
        )
        d = s.to_dict()
        restored = PipelineState.from_dict(d)
        assert restored.encoder_outs == s.encoder_outs

    def test_omni_event_types(self):
        from sglang_omni.models.minicpm_v.io import OmniEvent

        # All event types should be constructable
        for etype in ("text_delta", "text_final", "audio_delta", "audio_final", "debug", "final"):
            event = OmniEvent(type=etype, modality="text", payload={})
            assert event.type == etype


# ---------------------------------------------------------------------------
# preprocessor.py  — build_minicpm_image_mm_inputs
# ---------------------------------------------------------------------------


class TestBuildMiniCPMImageMMInputs:
    """Unit tests for build_minicpm_image_mm_inputs without loading real model."""

    def test_empty_hf_inputs(self):
        from sglang_omni.models.minicpm_v.components.preprocessor import (
            build_minicpm_image_mm_inputs,
        )

        result = build_minicpm_image_mm_inputs({})
        assert "pixel_values" in result
        assert result["slice_lengths"] == []

    def test_list_of_tensors_concatenated(self):
        from sglang_omni.models.minicpm_v.components.preprocessor import (
            build_minicpm_image_mm_inputs,
        )

        # Simulate two images: first with 3 slices, second with 2 slices
        pv1 = torch.randn(3, 3, 224, 224)
        pv2 = torch.randn(2, 3, 224, 224)
        tgt = torch.tensor([[16, 16], [16, 16], [16, 16], [8, 8], [8, 8]])

        result = build_minicpm_image_mm_inputs(
            {"pixel_values": [pv1, pv2], "tgt_sizes": tgt}
        )

        assert isinstance(result["pixel_values"], torch.Tensor)
        assert result["pixel_values"].shape[0] == 5  # 3 + 2 slices concatenated
        assert result["slice_lengths"] == [3, 2]
        assert torch.equal(result["tgt_sizes"], tgt)

    def test_single_tensor_becomes_list(self):
        from sglang_omni.models.minicpm_v.components.preprocessor import (
            build_minicpm_image_mm_inputs,
        )

        pv = torch.randn(4, 3, 224, 224)
        result = build_minicpm_image_mm_inputs({"pixel_values": pv})

        assert result["slice_lengths"] == [4]
        assert torch.equal(result["pixel_values"], pv)

    def test_image_sizes_forwarded(self):
        from sglang_omni.models.minicpm_v.components.preprocessor import (
            build_minicpm_image_mm_inputs,
        )

        pv = torch.randn(1, 3, 224, 224)
        img_sizes = [[224, 224]]
        result = build_minicpm_image_mm_inputs(
            {"pixel_values": pv, "image_sizes": img_sizes}
        )
        assert result["image_sizes"] == img_sizes


# ---------------------------------------------------------------------------
# merge.py — build_llm_inputs
# ---------------------------------------------------------------------------


class TestBuildLLMInputs:
    """Tests for build_llm_inputs with various encoder output configurations."""

    def _make_state(self, mm_inputs: dict | None = None) -> Any:
        from sglang_omni.models.minicpm_v.io import PipelineState

        return PipelineState(mm_inputs=mm_inputs or {})

    def test_empty_encoder_outs_returns_empty_model_inputs(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import build_llm_inputs

        state = self._make_state()
        result = build_llm_inputs(state, {})
        assert result["model_inputs"] == {}

    def test_image_embeds_forwarded(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import build_llm_inputs

        img_embeds = torch.randn(10, 512)
        tgt_sizes = torch.tensor([[8, 8], [8, 8]])

        encoder_outs = {
            "image_encoder": {
                "image_embeds": img_embeds,
                "tgt_sizes": tgt_sizes,
                "slice_lengths": [2],
            }
        }
        state = self._make_state({"image": {}})
        result = build_llm_inputs(state, encoder_outs)

        mi = result["model_inputs"]
        assert "image_embeds" in mi
        assert torch.equal(mi["image_embeds"], img_embeds)
        assert torch.equal(mi["tgt_sizes"], tgt_sizes)
        assert mi["slice_lengths"] == [2]

    def test_audio_embeds_forwarded(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import build_llm_inputs

        audio_embeds = torch.randn(30, 256)
        audio_lengths = [30]

        encoder_outs = {
            "audio_encoder": {
                "audio_embeds": audio_embeds,
                "audio_output_lengths": audio_lengths,
            }
        }
        state = self._make_state({"audio": {}})
        result = build_llm_inputs(state, encoder_outs)

        mi = result["model_inputs"]
        assert "audio_embeds" in mi
        assert torch.equal(mi["audio_embeds"], audio_embeds)
        assert mi["audio_output_lengths"] == audio_lengths

    def test_image_and_audio_together(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import build_llm_inputs

        img_embeds = torch.randn(5, 512)
        audio_embeds = torch.randn(20, 256)

        encoder_outs = {
            "image_encoder": {"image_embeds": img_embeds, "tgt_sizes": [[8, 8]]},
            "audio_encoder": {"audio_embeds": audio_embeds, "audio_output_lengths": [20]},
        }
        state = self._make_state({"image": {}, "audio": {}})
        result = build_llm_inputs(state, encoder_outs)

        mi = result["model_inputs"]
        assert "image_embeds" in mi
        assert "audio_embeds" in mi

    def test_tgt_sizes_falls_back_to_mm_inputs(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import build_llm_inputs

        tgt_sizes = torch.tensor([[4, 4]])
        state = self._make_state({"image": {"tgt_sizes": tgt_sizes}})
        # No image_encoder in encoder_outs, but tgt_sizes in mm_inputs
        result = build_llm_inputs(state, {})
        mi = result["model_inputs"]
        # tgt_sizes should be picked up from mm_inputs fallback
        assert "tgt_sizes" in mi

    def test_none_encoder_outs_for_audio_skipped_cleanly(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import build_llm_inputs

        img_embeds = torch.randn(5, 512)
        encoder_outs = {
            "image_encoder": {"image_embeds": img_embeds, "tgt_sizes": [[4, 4]]},
            # audio_encoder absent
        }
        state = self._make_state({"image": {}, "audio": {}})
        result = build_llm_inputs(state, encoder_outs)
        mi = result["model_inputs"]
        assert "audio_embeds" not in mi
        assert "image_embeds" in mi


# ---------------------------------------------------------------------------
# merge.py — _prune_preprocessing_for_llm
# ---------------------------------------------------------------------------


class TestPrunePreprocessing:
    """_prune_preprocessing_for_llm should retain minimal metadata."""

    def test_image_only_retains_tgt_sizes(self):
        from sglang_omni.models.minicpm_v.io import PipelineState
        from sglang_omni.models.minicpm_v.pipeline.merge import _prune_preprocessing_for_llm

        tgt = torch.tensor([[8, 8]])
        state = PipelineState(
            mm_inputs={
                "image": {
                    "pixel_values": torch.randn(1, 3, 224, 224),
                    "tgt_sizes": tgt,
                    "slice_lengths": [1],
                }
            }
        )
        _prune_preprocessing_for_llm(state, {"image_encoder": {"tgt_sizes": tgt}})
        # pixel_values should have been pruned
        assert "pixel_values" not in state.mm_inputs.get("image", {})
        assert "tgt_sizes" in state.mm_inputs.get("image", {})

    def test_audio_metadata_preserved(self):
        from sglang_omni.models.minicpm_v.io import PipelineState
        from sglang_omni.models.minicpm_v.pipeline.merge import _prune_preprocessing_for_llm

        state = PipelineState(mm_inputs={"image": {}, "audio": {}})
        encoder_outs = {
            "audio_encoder": {"audio_output_lengths": [16]},
        }
        _prune_preprocessing_for_llm(state, encoder_outs)
        assert state.mm_inputs.get("audio", {}).get("audio_output_lengths") == [16]

    def test_no_audio_encoder_out_no_audio_in_mm_inputs(self):
        from sglang_omni.models.minicpm_v.io import PipelineState
        from sglang_omni.models.minicpm_v.pipeline.merge import _prune_preprocessing_for_llm

        state = PipelineState(mm_inputs={"image": {}})
        _prune_preprocessing_for_llm(state, {})
        assert "audio" not in state.mm_inputs


# ---------------------------------------------------------------------------
# merge.py — decode_events
# ---------------------------------------------------------------------------


class TestDecodeEvents:
    """decode_events should produce OmniEvent objects for streaming and final output."""

    def _make_tokenizer(self, mapping: dict[int, str] | None = None) -> Any:
        """Build a minimal stub tokenizer."""
        mapping = mapping or {
            1: "Hello",
            2: " world",
            3: "!",
        }

        class _FakeTokenizer:
            eos_token_id = 99

            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(mapping.get(t, "") for t in token_ids)

        return _FakeTokenizer()

    def _make_state(self):
        from sglang_omni.models.minicpm_v.io import PipelineState

        return PipelineState(stream_state={"token_ids": [], "text": "", "emitted_text": ""})

    def test_is_final_produces_text_final_event(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import decode_events

        tokenizer = self._make_tokenizer()
        state = self._make_state()
        llm_out = {"output_ids": [1, 2], "is_final": True, "step": 2}

        events = list(
            decode_events(
                llm_out=llm_out,
                state=state,
                tokenizer=tokenizer,
                eos_token_id=99,
                step=2,
            )
        )
        assert len(events) == 1
        assert events[0].type == "text_final"
        assert events[0].is_final is True
        assert "Hello world" in events[0].payload["text"]

    def test_eos_token_terminates_stream(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import decode_events

        tokenizer = self._make_tokenizer()
        state = self._make_state()
        llm_out = {"output_ids": [99], "is_final": False, "step": 1}

        events = list(
            decode_events(
                llm_out=llm_out,
                state=state,
                tokenizer=tokenizer,
                eos_token_id=99,
                step=1,
            )
        )
        assert len(events) == 1
        assert events[0].type == "text_final"
        assert events[0].is_final is True

    def test_streaming_delta_events(self):
        from sglang_omni.models.minicpm_v.pipeline.merge import decode_events

        tokenizer = self._make_tokenizer({1: "Hello", 2: " world"})
        state = self._make_state()

        # Step 1: emit token 1
        events1 = list(
            decode_events(
                llm_out={"output_ids": [1], "is_final": False, "step": 1},
                state=state,
                tokenizer=tokenizer,
                eos_token_id=99,
                step=1,
            )
        )
        assert any(e.type == "text_delta" for e in events1)

        # Step 2: emit token 2
        events2 = list(
            decode_events(
                llm_out={"output_ids": [1, 2], "is_final": False, "step": 2},
                state=state,
                tokenizer=tokenizer,
                eos_token_id=99,
                step=2,
            )
        )
        assert any(e.type == "text_delta" for e in events2)
        # Delta should be " world", not "Hello world"
        delta_text = next(
            e.payload.get("text", "") for e in events2 if e.type == "text_delta"
        )
        assert " world" in delta_text


# ---------------------------------------------------------------------------
# next_stage.py — routing logic
# ---------------------------------------------------------------------------


class TestStageRouting:
    """Stage routing functions."""

    def _make_payload(self, data: dict) -> Any:
        from sglang_omni.proto import StagePayload

        payload = MagicMock(spec=StagePayload)
        payload.data = data
        return payload

    def _make_preprocessing_state(
        self,
        encoder_inputs: dict,
    ) -> dict:
        from sglang_omni.models.minicpm_v.io import PipelineState

        return PipelineState(encoder_inputs=encoder_inputs).to_dict()

    # --- preprocessing_next ---

    def test_preprocessing_next_non_payload_returns_aggregate(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import preprocessing_next

        result = preprocessing_next("req-1", "not a payload")
        assert result == ["mm_aggregate"]

    def test_preprocessing_next_with_image_only(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import preprocessing_next

        data = self._make_preprocessing_state(
            {
                "image_encoder": {"pixel_values": "..."},
                "audio_encoder": {"_skip": True, "_result": {}},
            }
        )
        payload = self._make_payload(data)
        result = preprocessing_next("req-1", payload)

        assert "image_encoder" in result
        assert "mm_aggregate" in result
        assert "audio_encoder" not in result

    def test_preprocessing_next_with_image_and_audio(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import preprocessing_next

        data = self._make_preprocessing_state(
            {
                "image_encoder": {"pixel_values": "..."},
                "audio_encoder": {"input_features": "..."},
            }
        )
        payload = self._make_payload(data)
        result = preprocessing_next("req-1", payload)

        assert "image_encoder" in result
        assert "audio_encoder" in result
        assert "mm_aggregate" in result
        # aggregate is always last
        assert result[-1] == "mm_aggregate"

    def test_preprocessing_next_ordering_is_deterministic(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import preprocessing_next

        data = self._make_preprocessing_state(
            {
                "image_encoder": {"pixel_values": "..."},
                "audio_encoder": {"input_features": "..."},
            }
        )
        payload = self._make_payload(data)
        r1 = preprocessing_next("req-1", payload)
        r2 = preprocessing_next("req-1", payload)
        assert r1 == r2

    # --- encoder_next ---

    def test_encoder_next_always_returns_aggregate(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import encoder_next

        assert encoder_next("req-1", None) == "mm_aggregate"

    # --- aggregate_next ---

    def test_aggregate_next_returns_llm(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import aggregate_next

        assert aggregate_next("req-1", None) == "llm"

    # --- llm_next ---

    def test_llm_next_text_only_returns_decode(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import llm_next

        assert llm_next("req-1", None) == "decode"

    def test_llm_next_text_only_payload_returns_decode(self):
        from sglang_omni.models.minicpm_v.io import PipelineState
        from sglang_omni.models.minicpm_v.pipeline.next_stage import llm_next

        state = PipelineState(llm_out={"output_ids": [1, 2, 3], "is_final": True})
        payload = self._make_payload(state.to_dict())
        result = llm_next("req-1", payload)
        assert result == "decode"

    def test_llm_next_audio_output_flag_routes_to_vocoder(self):
        from sglang_omni.models.minicpm_v.io import PipelineState
        from sglang_omni.models.minicpm_v.pipeline.next_stage import llm_next

        state = PipelineState(
            llm_out={
                "output_ids": [1, 2, 3],
                "is_final": True,
                "extra_model_outputs": {"has_audio_output": True},
            }
        )
        payload = self._make_payload(state.to_dict())
        result = llm_next("req-1", payload)
        assert isinstance(result, list)
        assert "vocoder" in result
        assert "decode" in result

    def test_llm_next_audio_output_modality_param_routes_to_vocoder(self):
        from sglang_omni.models.minicpm_v.io import PipelineState
        from sglang_omni.models.minicpm_v.pipeline.next_stage import llm_next

        state = PipelineState(
            raw_inputs={"params": {"output_modality": "audio"}},
            llm_out={"output_ids": [1, 2, 3], "is_final": True},
        )
        payload = self._make_payload(state.to_dict())
        result = llm_next("req-1", payload)
        assert isinstance(result, list)
        assert "vocoder" in result

    # --- vocoder_next ---

    def test_vocoder_next_returns_decode(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import vocoder_next

        assert vocoder_next("req-1", None) == "decode"

    # --- decode_next ---

    def test_decode_next_returns_none(self):
        from sglang_omni.models.minicpm_v.pipeline.next_stage import decode_next

        assert decode_next("req-1", None) is None


# ---------------------------------------------------------------------------
# config.py — pipeline config structure
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Validate pipeline config stage names and routing."""

    def test_minicpmv_config_stage_names(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMVPipelineConfig

        cfg = MiniCPMVPipelineConfig(model_path="dummy")
        names = [s.name for s in cfg.stages]
        assert names == ["preprocessing", "image_encoder", "mm_aggregate", "llm", "decode"]

    def test_minicpmv_hf_config_stage_names(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMVHFPipelineConfig

        cfg = MiniCPMVHFPipelineConfig(model_path="dummy")
        names = [s.name for s in cfg.stages]
        assert names == ["preprocessing", "image_encoder", "mm_aggregate", "llm", "decode"]

    def test_minicpmo_config_stage_names(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMOPipelineConfig

        cfg = MiniCPMOPipelineConfig(model_path="dummy")
        names = [s.name for s in cfg.stages]
        assert names == [
            "preprocessing",
            "image_encoder",
            "audio_encoder",
            "mm_aggregate",
            "llm",
            "vocoder",
            "decode",
        ]

    def test_minicpmv_entry_class(self):
        from sglang_omni.models.minicpm_v.config import (
            EntryClass,
            MiniCPMVPipelineConfig,
        )

        assert EntryClass is MiniCPMVPipelineConfig

    def test_minicpmv_architecture(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMVPipelineConfig

        assert MiniCPMVPipelineConfig.architecture == "MiniCPMV"

    def test_minicpmo_architecture(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMOPipelineConfig

        assert MiniCPMOPipelineConfig.architecture == "MiniCPMO"

    def test_minicpmo_aggregate_sources_include_audio(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMOPipelineConfig

        cfg = MiniCPMOPipelineConfig(model_path="dummy")
        agg = next(s for s in cfg.stages if s.name == "mm_aggregate")
        assert "audio_encoder" in agg.input_handler.sources

    def test_minicpmv_aggregate_sources_no_audio(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMVPipelineConfig

        cfg = MiniCPMVPipelineConfig(model_path="dummy")
        agg = next(s for s in cfg.stages if s.name == "mm_aggregate")
        assert "audio_encoder" not in agg.input_handler.sources

    def test_llm_stage_uses_sglang_factory(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMVPipelineConfig

        cfg = MiniCPMVPipelineConfig(model_path="dummy")
        llm = next(s for s in cfg.stages if s.name == "llm")
        assert "sglang" in llm.executor.factory.lower()

    def test_minicpmo_vocoder_stage_factory(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMOPipelineConfig

        cfg = MiniCPMOPipelineConfig(model_path="dummy")
        vocoder = next(s for s in cfg.stages if s.name == "vocoder")
        assert "vocoder" in vocoder.executor.factory.lower()

    def test_minicpmo_llm_next_is_dynamic(self):
        from sglang_omni.models.minicpm_v.config import MiniCPMOPipelineConfig

        cfg = MiniCPMOPipelineConfig(model_path="dummy")
        llm = next(s for s in cfg.stages if s.name == "llm")
        # llm_next supports dynamic routing (text vs audio output)
        assert "llm_next" in llm.get_next


# ---------------------------------------------------------------------------
# merge.py — merge_for_llm integration
# ---------------------------------------------------------------------------


class TestMergeForLLM:
    """merge_for_llm should correctly combine multi-payload stage outputs."""

    def _make_stage_payload(self, data: dict) -> Any:
        from sglang_omni.proto import StagePayload

        payload = MagicMock(spec=StagePayload)
        payload.data = data
        payload.request_id = "req-test"
        return payload

    def test_merge_with_image_encoder_output(self):
        from sglang_omni.models.minicpm_v.io import PipelineState
        from sglang_omni.models.minicpm_v.pipeline.merge import merge_for_llm

        img_embeds = torch.randn(4, 512)
        tgt_sizes = torch.tensor([[8, 8], [8, 8]])

        # preprocessing payload
        pre_state = PipelineState(
            prompt={"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "prompt_text": "hi"},
            mm_inputs={"image": {}},
        )
        pre_payload = self._make_stage_payload(pre_state.to_dict())

        # image_encoder payload
        enc_state = PipelineState(
            engine_outputs={
                "image_encoder": {
                    "image_embeds": img_embeds.tolist(),
                    "tgt_sizes": tgt_sizes.tolist(),
                }
            }
        )
        enc_payload = self._make_stage_payload(enc_state.to_dict())

        result = merge_for_llm(
            {"preprocessing": pre_payload, "image_encoder": enc_payload}
        )

        merged = PipelineState.from_dict(result.data)
        mi = merged.llm_inputs.get("model_inputs", {})
        assert "image_embeds" in mi or "tgt_sizes" in mi

    def test_merge_preserves_prompt(self):
        from sglang_omni.models.minicpm_v.io import PipelineState
        from sglang_omni.models.minicpm_v.pipeline.merge import merge_for_llm

        prompt = {"input_ids": [10, 20], "attention_mask": [1, 1], "prompt_text": "test"}
        pre_state = PipelineState(prompt=prompt, mm_inputs={"image": {}})
        pre_payload = self._make_stage_payload(pre_state.to_dict())

        result = merge_for_llm({"preprocessing": pre_payload})
        merged = PipelineState.from_dict(result.data)
        assert merged.prompt == prompt


# ---------------------------------------------------------------------------
# vocoder/cosyvoice.py — unit tests without loading actual model
# ---------------------------------------------------------------------------


class TestCosyVoiceVocoder:
    """Unit tests for CosyVoiceVocoder helper methods (no GPU required)."""

    def test_extract_audio_tokens_no_markers(self):
        from sglang_omni.models.minicpm_v.vocoder.cosyvoice import CosyVoiceVocoder

        # Build a minimal stub that avoids loading real model
        with patch.object(
            CosyVoiceVocoder, "__init__", lambda self, *a, **kw: None
        ):
            v = CosyVoiceVocoder.__new__(CosyVoiceVocoder)
            v._stream_cache = {}

        tokens = torch.tensor([10, 20, 30])
        result = v._extract_audio_tokens(tokens)
        assert torch.equal(result, tokens)

    def test_extract_audio_tokens_with_markers(self):
        from sglang_omni.models.minicpm_v.vocoder.cosyvoice import CosyVoiceVocoder

        with patch.object(
            CosyVoiceVocoder, "__init__", lambda self, *a, **kw: None
        ):
            v = CosyVoiceVocoder.__new__(CosyVoiceVocoder)
            v._stream_cache = {}

        # [text, audio_start, audio1, audio2, audio_end, text]
        tokens = torch.tensor([1, 100, 200, 201, 101, 2])
        result = v._extract_audio_tokens(tokens, audio_start_id=100, audio_end_id=101)
        expected = torch.tensor([200, 201])
        assert torch.equal(result, expected)

    def test_extract_audio_tokens_empty_returns_empty(self):
        from sglang_omni.models.minicpm_v.vocoder.cosyvoice import CosyVoiceVocoder

        with patch.object(
            CosyVoiceVocoder, "__init__", lambda self, *a, **kw: None
        ):
            v = CosyVoiceVocoder.__new__(CosyVoiceVocoder)
            v._stream_cache = {}

        tokens = torch.tensor([1, 2, 3])  # no audio markers
        result = v._extract_audio_tokens(tokens, audio_start_id=100, audio_end_id=101)
        assert result is None or (isinstance(result, torch.Tensor) and result.numel() == 0)

    def test_reset_stream_clears_cache(self):
        from sglang_omni.models.minicpm_v.vocoder.cosyvoice import CosyVoiceVocoder

        with patch.object(
            CosyVoiceVocoder, "__init__", lambda self, *a, **kw: None
        ):
            v = CosyVoiceVocoder.__new__(CosyVoiceVocoder)
            v._stream_cache = {"req-1": {"tokens": [1, 2, 3]}}

        v.reset_stream("req-1")
        assert "req-1" not in v._stream_cache


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
