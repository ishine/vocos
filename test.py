import os
import random

import soundfile
import torch

from vocos import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures



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



# vocos = Vocos.from_pretrained('pretrain/config.yaml', 'pretrain/pytorch_model.bin')
import torchaudio
wavpath = "../../Downloads/VO_paimon/vo_LYAQ206_2_paimon_04.wav"

y, sr = torchaudio.load(wavpath)
print(wavpath)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
feature_extractor = MelSpectrogramFeatures()
feat = feature_extractor(y)
#
plot_spectrogram_to_numpy(feat[0])
#
# print(feat.shape)
# y_hat = vocos.decode(feat)
#
# soundfile.write(wavpath+"rec5.wav",y_hat.numpy()[0,:], 24000)
# print(y_hat.shape)
#
