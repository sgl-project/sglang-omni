import math

import torch
from einops import einsum, rearrange
from torch import nn

from sglang_omni.models.nemotron_voicechat.preprocess import (
    DEFAULT_LOG_ZERO_GUARD,
    DEFAULT_PREEMPHASIS,
    LogMelFeatures,
)

SAMPLES_PER_FRAME = 1_280
SUBSAMPLING_KERNEL_SIZE = 3
SUBSAMPLING_STRIDE = 2
SUBSAMPLING_CACHE_SIZE = SUBSAMPLING_KERNEL_SIZE - 1
POINTWISE_CONV_KERNEL_SIZE = 1
FEED_FORWARD_RESIDUAL_SCALE = 0.5


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, *, groups=1):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=SUBSAMPLING_KERNEL_SIZE,
            stride=SUBSAMPLING_STRIDE,
            padding=0,
            groups=groups,
        )
        self.causal_padding = (SUBSAMPLING_KERNEL_SIZE - 1, SUBSAMPLING_STRIDE - 1)


class ConvSubsampling(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_mels = int(config["feat_in"])
        hidden_size = int(config["d_model"])
        channels = int(config["subsampling_conv_channels"])
        factor = int(config["subsampling_factor"])
        num_stages = int(math.log2(factor))

        subsampling_layers: list[nn.Module] = [
            CausalConv2d(in_channels=1, out_channels=channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_stages - 1):
            subsampling_layers.extend(
                (
                    CausalConv2d(
                        in_channels=channels, out_channels=channels, groups=channels
                    ),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                    ),
                    nn.ReLU(inplace=True),
                )
            )

        self.conv = nn.Sequential(*subsampling_layers)
        left_padding = SUBSAMPLING_KERNEL_SIZE - 1
        right_padding = SUBSAMPLING_STRIDE - 1
        for _ in range(num_stages):
            num_mels = (
                num_mels + left_padding + right_padding - SUBSAMPLING_KERNEL_SIZE
            ) // SUBSAMPLING_STRIDE + 1
        self.out = nn.Linear(in_features=channels * num_mels, out_features=hidden_size)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, *, kernel_size, groups=1, bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=groups,
            bias=bias,
        )
        self.left_padding = kernel_size - 1


class ConformerConvolution(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        kernel_size = int(config["conv_kernel_size"])
        use_bias = bool(config["use_bias"])
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=2 * hidden_size,
            kernel_size=POINTWISE_CONV_KERNEL_SIZE,
            bias=use_bias,
        )
        self.depthwise_conv = CausalConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=use_bias,
        )
        # Name is misleading, it's not batch norm, but layer norm
        self.batch_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=POINTWISE_CONV_KERNEL_SIZE,
            bias=use_bias,
        )


class ConformerFeedForward(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        expansion = int(config["ff_expansion_factor"])
        intermediate_size = int(hidden_size * expansion)
        use_bias = bool(config["use_bias"])
        self.linear1 = nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, bias=use_bias
        )
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, bias=use_bias
        )

    def forward(self, hidden_BTD):
        expanded_BTE = self.linear1(hidden_BTD)
        expanded_BTE = self.activation(expanded_BTE)
        hidden_BTD = self.linear2(expanded_BTE)
        return hidden_BTD


class RelPositionalEncoding(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        inverse_freq_H = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * -(math.log(10000.0) / hidden_size)
        )
        self.register_buffer("inverse_freq_H", inverse_freq_H, persistent=False)


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        self.num_heads = int(config["n_heads"])
        self.head_size = hidden_size // self.num_heads
        use_bias = bool(config["use_bias"])
        self.left_context, self.right_context = (
            int(value) for value in config["att_context_size"]
        )

        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_out = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))


class ConformerLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        self.norm_feed_forward1 = nn.LayerNorm(hidden_size)
        self.feed_forward1 = ConformerFeedForward(config)
        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.self_attn = RelPositionMultiHeadAttention(config)
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.conv = ConformerConvolution(config)
        self.norm_feed_forward2 = nn.LayerNorm(hidden_size)
        self.feed_forward2 = ConformerFeedForward(config)
        self.norm_out = nn.LayerNorm(hidden_size)


class ConformerEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.pre_encode = ConvSubsampling(config)
        self.pos_enc = RelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            ConformerLayer(config) for _ in range(int(config["n_layers"]))
        )
        self.xscaling = bool(config["xscaling"])


class AudioPerception(nn.Module):
    """Weights container plus the one inference path: streaming.

    Whole-utterance encoding is the streaming loop run to completion — the
    trailing row the causal right-padding used to produce comes from flush().
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.preprocessor = LogMelFeatures(config["preprocessor"])
        self.encoder = ConformerEncoder(config["encoder"])
        self.proj = nn.Linear(
            int(config["encoder"]["d_model"]), int(config["output_dim"])
        )

    def forward(self, waveform_BL):
        assert waveform_BL.shape[0] == 1
        assert waveform_BL.shape[1] % SAMPLES_PER_FRAME == 0
        stream = StreamingPerception(self)
        rows = [stream.push(block) for block in waveform_BL[0].split(SAMPLES_PER_FRAME)]
        rows.append(stream.flush())
        return torch.cat(rows).unsqueeze(0)


class StreamingPerception:
    """Incremental encoding, one 1280-sample block to one acoustic row.

    Attention keeps at most left_context+1 keys per layer, the convolutions
    keep their causal left windows, and the featurizer keeps the STFT overlap
    plus the preemphasis carry, so each push touches one frame of compute.
    """

    def __init__(self, perception: AudioPerception) -> None:
        self.perception = perception
        parameter = perception.proj.weight
        self.device = parameter.device
        self.dtype = parameter.dtype

        preprocessor = perception.preprocessor
        assert SAMPLES_PER_FRAME % preprocessor.hop_length == 0

        encoder = perception.encoder
        convs = list(encoder.pre_encode.conv)
        self.sub_stages = ((convs[0], None), (convs[2], convs[3]), (convs[5], convs[6]))
        self.xscale = (
            math.sqrt(encoder.layers[0].norm_out.weight.shape[0])
            if encoder.xscaling
            else 1.0
        )

        attention = encoder.layers[0].self_attn
        assert attention.right_context == 0
        self.max_keys = attention.left_context + 1
        self.num_heads = attention.num_heads
        self.head_size = attention.head_size

        with torch.inference_mode():
            offsets_P = torch.arange(
                self.max_keys - 1, -1, -1, device=self.device, dtype=torch.float32
            )
            angles_PH = einsum(offsets_P, encoder.pos_enc.inverse_freq_H, "p, h -> p h")
            encoding_PD = rearrange(
                torch.stack((angles_PH.sin(), angles_PH.cos()), dim=-1),
                "p h two -> p (h two)",
            ).to(self.dtype)
            # Row p holds the encoding for relative offset max_keys - 1 - p, so
            # row order matches cache order (oldest key first) once sliced to
            # the current number of keys.
            self.posproj_LPHS = [
                rearrange(
                    layer.self_attn.linear_pos(encoding_PD),
                    "p (h s) -> p h s",
                    h=self.num_heads,
                )
                for layer in encoder.layers
            ]

        self.reset()

    def reset(self) -> None:
        perception = self.perception
        preprocessor = perception.preprocessor
        self.preemphasis_carry = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.sample_buffer = torch.zeros(
            preprocessor.left_padding, device=self.device, dtype=self.dtype
        )

        num_mels = preprocessor.featurizer.fb.shape[1]
        sub_cache_shapes = []
        channels, freqs = 1, num_mels
        for conv, _ in self.sub_stages:
            sub_cache_shapes.append((1, channels, SUBSAMPLING_CACHE_SIZE, freqs))
            channels = conv.out_channels
            freqs = freqs // SUBSAMPLING_STRIDE + 1
        self.sub_caches = [
            torch.zeros(shape, device=self.device, dtype=self.dtype)
            for shape in sub_cache_shapes
        ]

        layers = perception.encoder.layers
        self.key_caches = [
            torch.zeros(
                0, self.num_heads, self.head_size, device=self.device, dtype=self.dtype
            )
            for _ in layers
        ]
        self.value_caches = [
            torch.zeros(
                0, self.num_heads, self.head_size, device=self.device, dtype=self.dtype
            )
            for _ in layers
        ]
        self.conv_caches = [
            torch.zeros(
                1,
                layer.conv.depthwise_conv.in_channels,
                layer.conv.depthwise_conv.left_padding,
                device=self.device,
                dtype=self.dtype,
            )
            for layer in layers
        ]
        self.flushed = False

    @torch.inference_mode()
    def push(self, samples_S: torch.Tensor) -> torch.Tensor:
        """One 1280-sample block in, one acoustic row out; call i yields
        encoder row i (the first call returns the leading pad row)."""
        assert not self.flushed
        assert samples_S.shape == (SAMPLES_PER_FRAME,)
        samples_S = samples_S.to(device=self.device, dtype=self.dtype)
        mel_1TM = self._featurize(samples_S)
        return self._encode(mel_1TM)

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        """The trailing row whole-utterance encoding gets from its causal
        right padding; live sessions never need it."""
        assert not self.flushed
        self.flushed = True
        num_mels = self.perception.preprocessor.featurizer.fb.shape[1]
        zero_mel_1TM = torch.zeros(1, 1, num_mels, device=self.device, dtype=self.dtype)
        return self._encode(zero_mel_1TM)

    def _featurize(self, samples_S: torch.Tensor) -> torch.Tensor:
        preprocessor = self.perception.preprocessor
        previous_S = torch.cat((self.preemphasis_carry, samples_S[:-1]))
        preemphasized_S = samples_S - DEFAULT_PREEMPHASIS * previous_S
        window_W = torch.cat((self.sample_buffer, preemphasized_S))
        self.sample_buffer = window_W[-preprocessor.left_padding :]
        self.preemphasis_carry = samples_S[-1:]

        spectrum_FT = torch.stft(
            rearrange(window_W, "w -> 1 w"),
            n_fft=preprocessor.n_fft,
            hop_length=preprocessor.hop_length,
            win_length=preprocessor.win_length,
            return_complex=True,
            center=False,
            window=preprocessor.featurizer.window,
        )[0]
        power_FT = spectrum_FT.abs().square()
        mel_MT = einsum(preprocessor.featurizer.fb[0], power_FT, "m f, f t -> m t")
        log_mel_MT = torch.log(mel_MT + DEFAULT_LOG_ZERO_GUARD)
        return rearrange(log_mel_MT, "m t -> 1 t m")

    def _encode(self, mel_1TM: torch.Tensor) -> torch.Tensor:
        hidden_11D = self._subsample(mel_1TM) * self.xscale
        for index, layer in enumerate(self.perception.encoder.layers):
            hidden_11D = hidden_11D + FEED_FORWARD_RESIDUAL_SCALE * layer.feed_forward1(
                layer.norm_feed_forward1(hidden_11D)
            )
            hidden_11D = hidden_11D + self._attend(
                index, layer.self_attn, layer.norm_self_att(hidden_11D)
            )
            hidden_11D = hidden_11D + self._convolve(
                index, layer.conv, layer.norm_conv(hidden_11D)
            )
            hidden_11D = hidden_11D + FEED_FORWARD_RESIDUAL_SCALE * layer.feed_forward2(
                layer.norm_feed_forward2(hidden_11D)
            )
            hidden_11D = layer.norm_out(hidden_11D)
        return self.perception.proj(hidden_11D)[0]

    def _subsample(self, mel_1TM: torch.Tensor) -> torch.Tensor:
        hidden_1CTM = rearrange(mel_1TM, "b t m -> b 1 t m")
        for stage, (conv, pointwise) in enumerate(self.sub_stages):
            hidden_1CTM = torch.cat((self.sub_caches[stage], hidden_1CTM), dim=2)
            self.sub_caches[stage] = hidden_1CTM[:, :, -SUBSAMPLING_CACHE_SIZE:]
            # The cache stands in for the module's causal time padding; only
            # the frequency axis still pads here.
            padded_1CTM = nn.functional.pad(hidden_1CTM, (*conv.causal_padding, 0, 0))
            hidden_1CTM = nn.Conv2d.forward(conv, padded_1CTM)
            if pointwise is not None:
                hidden_1CTM = pointwise(hidden_1CTM)
            hidden_1CTM = torch.relu(hidden_1CTM)
        return self.perception.encoder.pre_encode.out(
            rearrange(hidden_1CTM, "b c t m -> b t (c m)")
        )

    def _attend(self, index: int, attention, hidden_11D: torch.Tensor) -> torch.Tensor:
        query_HS = rearrange(
            attention.linear_q(hidden_11D), "1 1 (h s) -> h s", h=self.num_heads
        )
        key_HS = rearrange(
            attention.linear_k(hidden_11D), "1 1 (h s) -> h s", h=self.num_heads
        )
        value_HS = rearrange(
            attention.linear_v(hidden_11D), "1 1 (h s) -> h s", h=self.num_heads
        )
        keys_KHS = torch.cat((self.key_caches[index], key_HS[None]))[-self.max_keys :]
        values_KHS = torch.cat((self.value_caches[index], value_HS[None]))[
            -self.max_keys :
        ]
        self.key_caches[index] = keys_KHS
        self.value_caches[index] = values_KHS

        num_keys = keys_KHS.shape[0]
        content_KH = einsum(
            query_HS + attention.pos_bias_u, keys_KHS, "h s, k h s -> k h"
        )
        position_KH = einsum(
            query_HS + attention.pos_bias_v,
            self.posproj_LPHS[index][self.max_keys - num_keys :],
            "h s, k h s -> k h",
        )
        weights_KH = ((content_KH + position_KH) / math.sqrt(self.head_size)).softmax(
            dim=0
        )
        attended_HS = einsum(weights_KH, values_KHS, "k h, k h s -> h s")
        return attention.linear_out(rearrange(attended_HS, "h s -> 1 1 (h s)"))

    def _convolve(self, index: int, conv, hidden_11D: torch.Tensor) -> torch.Tensor:
        gates_1Gt = conv.pointwise_conv1(rearrange(hidden_11D, "b t d -> b d t"))
        hidden_1Dt = nn.functional.glu(gates_1Gt, dim=1)
        window_1DT = torch.cat((self.conv_caches[index], hidden_1Dt), dim=2)
        self.conv_caches[index] = window_1DT[:, :, 1:]
        hidden_1Dt = nn.Conv1d.forward(conv.depthwise_conv, window_1DT)
        hidden_11D = conv.batch_norm(rearrange(hidden_1Dt, "b d t -> b t d"))
        hidden_1Dt = conv.activation(rearrange(hidden_11D, "b t d -> b d t"))
        hidden_1Dt = conv.pointwise_conv2(hidden_1Dt)
        return rearrange(hidden_1Dt, "b d t -> b t d")
