import os
import random

import soundfile
import torch

from vocos import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures
import torchaudio

from vocos.mel_processing import load_wav_to_torch


def plot_spectrogram_to_numpy(spectrogram):

    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    return data

vocos = Vocos.from_pretrained('configs/vocos.yaml', '../../Downloads/pytorch_model (2).bin')
wavpath = "../../Downloads/vo_EQPM005_9_paimon_13.wav"

y, sr = load_wav_to_torch(wavpath, target_sr=44100)
feat = vocos.feature_extractor(y.unsqueeze(0))
plot_spectrogram_to_numpy(feat[0])

y_hat = vocos.decode(feat)

feat_hat = vocos.feature_extractor(y_hat)
plot_spectrogram_to_numpy(feat_hat[0])

soundfile.write(wavpath+"rec.wav",y_hat.numpy()[0,:], 44100)
