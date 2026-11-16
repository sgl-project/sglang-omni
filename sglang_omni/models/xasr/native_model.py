"""Original FP32 X-ASR streaming model with request-local and batched state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from sglang_omni.vendor.xasr.decoder import Decoder
from sglang_omni.vendor.xasr.joiner import Joiner
from sglang_omni.vendor.xasr.scaling_converter import convert_scaled_to_non_scaled

from .native_zipformer import StreamingZipformer


@dataclass
class StreamState:
    cache: list[torch.Tensor]
    tokens: list[int]
    decoder: torch.Tensor | None
    closed: bool = False
    chunks: int = 0


class NativeXASR(nn.Module):
    frame_count = 61
    frame_shift = 48

    def __init__(self, checkpoint, device="cuda"):
        super().__init__()
        path = Path(checkpoint)
        if path.is_dir():
            config = json.loads((path / "config.json").read_text())
            if (
                config.get("model_type") != "xasr_zipformer_transducer"
                or config.get("format_version") != 2
                or config.get("checkpoint_file") != "model.safetensors"
            ):
                raise ValueError("unsupported or invalid X-ASR model config")
            else:
                pass
            expected = config.get("checkpoint_sha256")
            path = path / "model.safetensors"
        else:
            expected = (
                "884663c45bdcb073eff3d356aa503cc5eb861dd42022e1b851acdbb14ca26467"
            )
        with path.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        if (
            sha != expected
            or sha != "884663c45bdcb073eff3d356aa503cc5eb861dd42022e1b851acdbb14ca26467"
        ):
            raise ValueError("model weights SHA-256 mismatch")
        else:
            pass
        source = load_file(str(path), device="cpu")
        self.encoder = StreamingZipformer()
        self.predictor = Decoder(
            vocab_size=5000, decoder_dim=512, blank_id=0, context_size=2
        )
        self.joiner = Joiner(
            encoder_dim=768, decoder_dim=512, joiner_dim=512, vocab_size=5000
        )
        target = self.encoder.state_dict()
        ew = {}
        for name, value in target.items():
            key = (
                "joiner.encoder_proj." + name[len("encoder_proj.") :]
                if name.startswith("encoder_proj.")
                else name
            )
            if key not in source or source[key].shape != value.shape:
                raise ValueError(f"missing or incompatible tensor {key}")
            else:
                pass
            ew[name] = source[key]
        self.encoder.load_state_dict(ew, strict=True)
        for prefix, module in [("decoder.", self.predictor), ("joiner.", self.joiner)]:
            subset = {
                name[len(prefix) :]: v
                for name, v in source.items()
                if name.startswith(prefix)
            }
            module.load_state_dict(subset, strict=True)
        del source, ew
        self.encoder.inference_conversion()
        convert_scaled_to_non_scaled(self.predictor, inplace=True, is_onnx=True)
        self.eval().to(device)
        self.batch_axes = [i % 6 in (0, 1, 2, 3) for i in range(114)] + [False, False]

    @property
    def device(self):
        return next(self.predictor.parameters()).device

    @torch.inference_mode()
    def new_stream(self):
        tokens = [0, 0]
        return StreamState(
            self.encoder.initial_states(1, self.device),
            tokens,
            self.predict_contexts([tokens]),
        )

    def predict_contexts(self, contexts):
        x = torch.tensor(contexts, dtype=torch.long, device=self.device)
        return self.joiner.decoder_proj(self.predictor(x, need_pad=False).squeeze(1))

    @torch.inference_mode()
    def step_batch(self, states, features):
        if not states or len(states) != len(features):
            raise ValueError("states and feature lengths differ")
        else:
            pass
        if len({id(s) for s in states}) != len(states):
            raise ValueError("duplicate stream")
        else:
            pass
        if any(s.closed for s in states):
            raise ValueError("stream closed")
        else:
            pass
        xs = [
            torch.as_tensor(x, dtype=torch.float32, device=self.device)
            for x in features
        ]
        if any(x.shape != (61, 80) or not torch.isfinite(x).all() for x in xs):
            raise ValueError("expected finite (61,80) features")
        else:
            pass
        cache = [
            torch.cat([s.cache[i] for s in states], dim=int(axis))
            for i, axis in enumerate(self.batch_axes)
        ]
        output, new_cache = self.encoder(torch.stack(xs), cache)
        next_cache = [
            [
                v.narrow(int(axis), j, 1).clone()
                for v, axis in zip(new_cache, self.batch_axes)
            ]
            for j in range(len(states))
        ]
        tokens = [s.tokens.copy() for s in states]
        decoder = torch.cat([s.decoder for s in states], dim=0)
        for frame in output.unbind(dim=1):
            ids = self.joiner(frame, decoder, project_input=False).argmax(-1).tolist()
            changed = [i for i, t in enumerate(ids) if t != 0]
            if changed:
                for i in changed:
                    tokens[i].append(ids[i])
                decoder = decoder.clone()
                decoder[changed] = self.predict_contexts(
                    [tokens[i][-2:] for i in changed]
                )
            else:
                pass
        new = [ids[len(s.tokens) :] for s, ids in zip(states, tokens)]
        for i, s in enumerate(states):
            s.cache = next_cache[i]
            s.tokens = tokens[i]
            s.decoder = decoder[i : i + 1].clone()
            s.chunks += 1
        return new

    def step(self, state, features):
        return self.step_batch([state], [features])[0]

    def close(self, state):
        state.closed = True
        state.cache.clear()
        state.decoder = None
