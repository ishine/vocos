from typing import Literal

import numpy as np
import scipy
import torch
from torch import nn, Tensor, ones, view_as_real, view_as_complex, hann_window
import torch.nn.functional as F


def rect_window(window_length: int, periodic: None | bool = True, *, dtype: None | torch.dtype = None, device: None | torch.device = None) -> Tensor:
    """Generate Rectangular window."""
    window = ones([window_length,])
    if dtype is not None:
        window = window.to(dtype)
    if device is not None:
        window = window.to(device)

    return window


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with windowing.
    This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: Literal["same", "center"] = "same", no_window: bool = False):
        """
        Args:
            n_fft      - Size of Fourier transform
            hop_length - The distance between neighboring sliding window frames
            win_length - The size of window frame and STFT filter
            padding    - Type of padding
        """
        super().__init__()

        # Validation
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")

        self.padding = padding
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length

        window = hann_window(win_length) if not no_window else rect_window(win_length)
        self.register_buffer("window", window)

    def forward(self, cspec: Tensor) -> Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            cspec :: (B, Freq, Frame) - Complex spectrogram
        Returns:
                  :: (B, T)           - Reconstructed time-domain signal
        """

        # [center] (PyTorch iSTFT, early return)
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(cspec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)

        # [same|causal]
        assert cspec.dim() == 3, "Expected a 3D tensor as input"
        n_frame = cspec.size()[2]

        ## spectrum-to-segment :: (B, Freq, Frame) -> (B, Segment, Frame) - iFT
        ifft = torch.fft.irfft(cspec, self.n_fft, dim=1, norm="backward")

        ## segments-to-wave :: (B, Segment, Frame) -> (B, T) - OverLap and Add
        ifft = ifft * self.window[None, :, None]
        output_size = (n_frame - 1) * self.hop_length + self.win_length
        y = F.fold(ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length))
        y = y[:, 0, 0, :]

        ## correction - Normalization by window envelope
        ### :: (Segment,) -> (B=1, Frame, Segment) -> (B=1, Segment, Frame)
        window_sq = self.window.square().expand(1, n_frame, -1).transpose(1, 2)
        ### :: (B=1, Segment, Frame) -> (1, 1, 1, T) -> (T,)
        window_envelope = F.fold(window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length)).squeeze()
        ### 'same' inverse padding
        pad = (self.win_length - self.hop_length) // 2
        y               =               y[:, pad:-pad]
        window_envelope = window_envelope[   pad:-pad]
        ### Check NOLA
        assert (window_envelope > 1e-11).all()
        ### Normalization
        y = y / window_envelope

        return y


class MDCT(nn.Module):
    """
    Modified Discrete Cosine Transform (MDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.cosine(frame_len)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(-1j * torch.pi * torch.arange(frame_len) / frame_len)
        post_twiddle = torch.exp(-1j * torch.pi * n0 * (torch.arange(N) + 0.5) / N)
        # view_as_real: NCCL Backend does not support ComplexFloat data type
        # https://github.com/pytorch/pytorch/issues/71613
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply the Modified Discrete Cosine Transform (MDCT) to the input audio.

        Args:
            audio (Tensor): Input audio waveform of shape (B, T), where B is the batch size
                and T is the length of the audio.

        Returns:
            Tensor: MDCT coefficients of shape (B, L, N), where L is the number of output frames
                and N is the number of frequency bins.
        """
        if self.padding == "center":
            audio = torch.nn.functional.pad(audio, (self.frame_len // 2, self.frame_len // 2))
        elif self.padding == "same":
            # hop_length is 1/2 frame_len
            audio = torch.nn.functional.pad(audio, (self.frame_len // 4, self.frame_len // 4))
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        x = audio.unfold(-1, self.frame_len, self.frame_len // 2)
        N = self.frame_len // 2
        x = x * self.window.expand(x.shape)
        X = torch.fft.fft(x * view_as_complex(self.pre_twiddle).expand(x.shape), dim=-1)[..., :N]
        res = X * view_as_complex(self.post_twiddle).expand(X.shape) * np.sqrt(1 / N)
        return torch.real(res) * np.sqrt(2)


class IMDCT(nn.Module):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.cosine(frame_len)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(1j * torch.pi * n0 * torch.arange(N * 2) / N)
        post_twiddle = torch.exp(1j * torch.pi * (torch.arange(N * 2) + n0) / (N * 2))
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the Inverse Modified Discrete Cosine Transform (IMDCT) to the input MDCT coefficients.

        Args:
            X (Tensor): Input MDCT coefficients of shape (B, L, N), where B is the batch size,
                L is the number of frames, and N is the number of frequency bins.

        Returns:
            Tensor: Reconstructed audio waveform of shape (B, T), where T is the length of the audio.
        """
        B, L, N = X.shape
        Y = torch.zeros((B, L, N * 2), dtype=X.dtype, device=X.device)
        Y[..., :N] = X
        Y[..., N:] = -1 * torch.conj(torch.flip(X, dims=(-1,)))
        y = torch.fft.ifft(Y * view_as_complex(self.pre_twiddle).expand(Y.shape), dim=-1)
        y = torch.real(y * view_as_complex(self.post_twiddle).expand(y.shape)) * np.sqrt(N) * np.sqrt(2)
        result = y * self.window.expand(y.shape)
        output_size = (1, (L + 1) * N)
        audio = torch.nn.functional.fold(
            result.transpose(1, 2),
            output_size=output_size,
            kernel_size=(1, self.frame_len),
            stride=(1, self.frame_len // 2),
        )[:, 0, 0, :]

        if self.padding == "center":
            pad = self.frame_len // 2
        elif self.padding == "same":
            pad = self.frame_len // 4
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        audio = audio[:, pad:-pad]
        return audio
