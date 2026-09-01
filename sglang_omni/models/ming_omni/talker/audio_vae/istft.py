import torch
import torch.nn as nn


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        self.audio_buffer = None
        self.window_buffer = None
        self.buffer_len = self.win_length - self.hop_length

    def __buffer_process(self, x, buffer, pad, last_chunk=False, streaming=False):
        if streaming:
            if buffer is None:
                # first chunk
                x = x[:, pad:]
            if buffer is not None:
                # next chunk
                x[:, : self.buffer_len] += buffer
            buffer = x[:, -self.buffer_len :]
            if not last_chunk:
                x = x[:, : -self.buffer_len]
            else:
                x = x[:, :-pad]
        else:
            x = x[:, pad:-pad]

        return x, buffer

    def overlap_add_components(
        self,
        spec: torch.Tensor,
        valid_frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the unnormalized audio numerator and window denominator."""
        assert spec.dim() == 3, "Expected a 3D tensor as input"
        _, _, frame_count = spec.shape

        if valid_frame_mask is not None:
            spec = spec * valid_frame_mask.unsqueeze(1)

        inverse = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        window = self.window
        inverse = inverse * window[None, :, None]

        output_size = (frame_count - 1) * self.hop_length + self.win_length
        numerator = torch.nn.functional.fold(
            inverse,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]

        window_frames = window.square().expand(1, frame_count, -1).transpose(1, 2)
        if valid_frame_mask is not None:
            window_frames = window_frames * valid_frame_mask.unsqueeze(1)
        denominator = torch.nn.functional.fold(
            window_frames,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]
        return numerator, denominator

    def forward(
        self,
        spec: torch.Tensor,
        audio_buffer=None,
        window_buffer=None,
        streaming=False,
        last_chunk=False,
    ):
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.
            audio_buffer (Tensor): [Streaming Input/State] The audio overlap buffer from the previous chunk.
                            Shape: (B, win_length - hop_length)
            window_buffer (Tensor): [Streaming Input/State] The window overlap buffer from the previous chunk.
            streaming: If `True`, the function operates in streaming mode, processing `spec` as a single chunk.
            last_chunk: When `streaming=True` and `last_chunk=True`, the function can perform final "flush" operations

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        y, window_envelope = self.overlap_add_components(spec)

        y, audio_buffer = self.__buffer_process(
            y, audio_buffer, pad, last_chunk=last_chunk, streaming=streaming
        )

        window_envelope, window_buffer = self.__buffer_process(
            window_envelope,
            window_buffer,
            pad,
            last_chunk=last_chunk,
            streaming=streaming,
        )
        window_envelope = window_envelope.squeeze()

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y, audio_buffer, window_buffer


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def predict_spectrum(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project hidden states into a complex spectrum and raw head output."""
        x_pred = self.out(x).transpose(1, 2)
        mag, phase = x_pred.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        spectrum = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        return spectrum, x_pred

    def forward(
        self,
        x: torch.Tensor,
        audio_buffer=None,
        window_buffer=None,
        streaming=False,
        last_chunk=False,
    ):
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        spectrum, x_pred = self.predict_spectrum(x)
        audio, audio_buffer, window_buffer = self.istft(
            spectrum,
            audio_buffer=audio_buffer,
            window_buffer=window_buffer,
            streaming=streaming,
            last_chunk=last_chunk,
        )
        return audio.unsqueeze(1), x_pred, audio_buffer, window_buffer
