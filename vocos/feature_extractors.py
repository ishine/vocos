from typing import List, Literal

import torch
from torch import nn, Tensor, hann_window
import torch.nn.functional as F
import torchaudio
from encodec import EncodecModel

from vocos.modules import safe_log
from vocos.spectral_ops import rect_window


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: Tensor, **kwargs) -> Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class SpectrogramFeatures(FeatureExtractor):
    """Wave-to-Spec feature extractor."""
    def __init__(self, sample_rate:int=24000, n_fft:int=1024, hop_length:int=256, n_mels:int=0, padding:Literal["center", "same", "causal"]="center", no_window: bool = False):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        # Validation
        if padding not in ["center", "same", "causal"]:
            raise ValueError("Padding must be 'center' or 'same' or 'causal'.")
        assert n_mels == 0, "n_mels is not supported now."

        self.padding = padding
        self.lin_spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            center=(padding=="center"), # Automatic center padding | Manual same/causal padding
            power=1,
            window_fn = hann_window if not no_window else rect_window,
        )

    def forward(self, audio, **kwargs):
        """Convert to linear-frequency log-magnitude spectrogram.
        
        Args:
            audio :: (B, T)
        Returns:
                  :: (B, Freq, Frame) - Spectrogram, Frame = T//hop ('same') | 
        """
        # Automatic center padding - Kernel axis align stride head, drop last
        if   self.padding == "center":
            pass
        # Manual same padding - Kernel axis align with stride center, drop last
        elif self.padding == "same":
            pad = self.lin_spec.win_length - self.lin_spec.hop_length
            half_pad = pad // 2
            audio = F.pad(audio, (half_pad, half_pad), mode="reflect")
        # Manual causal padding - DeltaKernel axis align with stride tail, drop last
        elif self.padding == "causal":
            pad = self.lin_spec.win_length - self.lin_spec.hop_length
            audio = F.pad(audio, (     pad,        0), mode="reflect")
        else:
            raise RuntimeError(f"Not supported padding type in wave-to-mel: {self.padding}")

        lin = self.lin_spec(audio)
        lin_freq_log_amp_spec = safe_log(lin)
        return lin_freq_log_amp_spec



class MelSpectrogramFeatures(FeatureExtractor):
    """Wave-to-Mel feature extractor."""
    def __init__(self, sample_rate:int=24000, n_fft:int=1024, hop_length:int=256, n_mels:int=100, padding:Literal["center", "same", "causal"]="center", no_window: bool = False):
        super().__init__()

        # Validation
        if padding not in ["center", "same", "causal"]:
            raise ValueError("Padding must be 'center' or 'same' or 'causal'.")

        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=(padding=="center"), # Automatic center padding | Manual same/causal padding
            power=1,
            window_fn = hann_window if not no_window else rect_window,
        )

    def forward(self, audio, **kwargs):
        """Convert to mel-frequency log-magnitude spectrogram.
        
        Args:
            audio :: (B, T)
        Returns:
                  :: (B, Freq, Frame) - Spectrogram, Frame = T//hop ('same') | 
        """
        # Automatic center padding - Kernel axis align stride head, drop last
        if   self.padding == "center":
            pass
        # Manual same padding - Kernel axis align with stride center, drop last
        elif self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            half_pad = pad // 2
            audio = F.pad(audio, (half_pad, half_pad), mode="reflect")
        # Manual causal padding - DeltaKernel axis align with stride tail, drop last
        elif self.padding == "causal":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = F.pad(audio, (     pad,        0), mode="reflect")
        else:
            raise RuntimeError(f"Not supported padding type in wave-to-mel: {self.padding}")

        mel = self.mel_spec(audio)
        mel_freq_log_amp_spec = safe_log(mel)
        return mel_freq_log_amp_spec


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
        no_window: bool = False,
    ):
        super().__init__()

        assert not no_window, "`no_window` is not supported yet."

        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat([vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]], dim=0)
        self.codebook_weights = torch.nn.Parameter(codebook_weights, requires_grad=train_codebooks)
        self.bandwidths = bandwidths

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, bandwidth_id: torch.Tensor):
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
        )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)
