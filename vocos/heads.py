from typing import Literal
import torch
from torch import nn, clip, exp, cos, sin
from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz

from vocos.spectral_ops import IMDCT, ISTFT
from vocos.modules import symexp
from vocos.domain import FeatSeries, Wave


class FourierHead(nn.Module):
    """ABC of inverse fourier modules."""

    def forward(self, x: FeatSeries) -> Wave:
        """
        Args:
            x - Feature series

        Returns:
              - Reconstructed time-domain audio
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTHead(FourierHead):
    """ISTFT Head for predicting audio waveform through complex spectrogram."""

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: Literal["center", "same", "causal"] = "same", no_window: bool = False):
        """
        Args:
            dim        - Feature dimension size of input feature series
            n_fft      - Size of Fourier transform
            hop_length - STFT hop
            padding    - Padding type specifier
        """
        super().__init__()

        # Dimension matching
        freq_x2 = n_fft + 2 # == 2 * (n_fft/2 + 1)
        self.out = torch.nn.Linear(dim, freq_x2)
        # complexSpec-to-wave
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding, no_window=no_window)

    def forward(self, x: FeatSeries) -> Wave:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x :: (B, Frame, Feat) - Feature series 
        Returns:
              :: (B, T)           - Reconstructed time-domain audio signal
        """
        # Dimension matching :: (B, Frame, Feat=i) -> (B, Frame, Feat=2*freq) -> (B, Feat=2*freq, Frame)
        x = self.out(x).transpose(1, 2)

        # feat-to-complexSpec :: (B, Feat=2*freq, Frame) -> 2 x (B, Freq, Frame) -> (B, Freq, Frame) - Magnitude (absolute value) scaling & Phase (argument) wrapping
        logabs, arg = x.chunk(2, dim=1)
        complex_spec = clip(exp(logabs), max=1e2) * (cos(arg) + 1j * sin(arg))

        # complexSpec-to-wave :: (B, Freq, Frame) -> (B, T) - iSTFT
        wave = self.istft(complex_spec)

        return wave


class IMDCTSymExpHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with symmetric exponential function

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        sample_rate (int, optional): The sample rate of the audio. If provided, the last layer will be initialized
                                     based on perceptual scaling. Defaults to None.
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(
        self, dim: int, mdct_frame_len: int, padding: str = "same", sample_rate: int = None, clip_audio: bool = False,
    ):
        super().__init__()
        out_dim = mdct_frame_len // 2
        self.out = nn.Linear(dim, out_dim)
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)
        self.clip_audio = clip_audio

        if sample_rate is not None:
            # optionally init the last layer following mel-scale
            m_max = _hz_to_mel(sample_rate // 2)
            m_pts = torch.linspace(0, m_max, out_dim)
            f_pts = _mel_to_hz(m_pts)
            scale = 1 - (f_pts / f_pts.max())

            with torch.no_grad():
                self.out.weight.mul_(scale.view(-1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IMDCTSymExpHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)
        x = symexp(x)
        x = torch.clip(x, min=-1e2, max=1e2)  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(x)
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)

        return audio


class IMDCTCosHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with parametrizing MDCT = exp(m) · cos(p)

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(self, dim: int, mdct_frame_len: int, padding: str = "same", clip_audio: bool = False):
        super().__init__()
        self.clip_audio = clip_audio
        self.out = nn.Linear(dim, mdct_frame_len)
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IMDCTCosHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)
        m, p = x.chunk(2, dim=2)
        m = torch.exp(m).clip(max=1e2)  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(m * torch.cos(p))
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)
        return audio
