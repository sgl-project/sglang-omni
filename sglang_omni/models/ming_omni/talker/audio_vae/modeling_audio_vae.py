import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_oobleck import (
    OobleckDiagonalGaussianDistribution,
)
from transformers import PreTrainedModel

from .configuration_audio_vae import AudioVAEconfig
from .vae_modules import Decoder, Encoder


class AudioVAE(PreTrainedModel):
    config_class = AudioVAEconfig

    def __init__(self, config: AudioVAEconfig):
        super().__init__(config)
        self.encoder = Encoder(
            encoder_args=config.enc_kwargs["backbone"],
            input_dim=config.enc_kwargs["input_dim"],
            hop_size=config.enc_kwargs.get("hop_size", 320),
            latent_dim=config.enc_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )

        self.decoder = Decoder(
            decoder_args=config.dec_kwargs["backbone"],
            output_dim=config.dec_kwargs["output_dim"],
            latent_dim=config.dec_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )

        self.post_init()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if self.config.init_method == "kaiming":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            else:
                module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def encode_latent(self, waveform, waveform_length):
        """
        Encodes a raw waveform to obtain its acoustic latent representation.
        Args:
            waveform: The input audio waveform, shape (B, T_wav).
            waveform_length: The length of each waveform, shape (B,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - z (torch.Tensor): The sampled acoustic latent vectors, shape (B, T_frame, D_latent).
                - frame_num (torch.Tensor): The number of frames for each audio, shape (B,).
        """
        frame_num = torch.ceil(
            waveform_length / self.config.enc_kwargs["input_dim"]
        ).to(torch.int32)
        if self.config.patch_size != -1:
            frame_num = torch.ceil(frame_num / self.config.patch_size)
        h, y = self.encoder(waveform)
        h = h.transpose(1, 2)  # [B, d, T]

        posterior = OobleckDiagonalGaussianDistribution(h)
        latent = posterior.sample()  # [B, d/2, T]
        latent = latent.transpose(1, 2)
        return latent, frame_num

    def decode(
        self,
        latent,
        past_key_values=None,
        use_cache=False,
        stream_state=(None, None, None),
        last_chunk=False,
    ):
        """
        Reconstructs the raw audio waveform from its acoustic latent representation.
        Args:
            latent: The acoustic latent representation, shape: (B, T_frame, D_latent).

        Returns:
            The reconstructed raw audio waveform, shape: (B, T_wav)
        """
        waveform, stream_state, past_key_values = self.decoder.low_level_reconstruct(
            latent,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stream_state=stream_state,
            last_chunk=last_chunk,
        )
        return waveform, stream_state, past_key_values
