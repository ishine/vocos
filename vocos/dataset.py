from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples:   int
    batch_size:    int
    num_workers:   int
    cache_cuda:    bool = False


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config   = val_params
        
        self._train_loader: None | DataLoader = None
        self._val_loader:   None | DataLoader = None

    def _set_dataloder(self, cfg: DataConfig, train: bool):
        if train and self._train_loader is None:
            dataset = VocosDataset(cfg, train=train)
            self._train_loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True, persistent_workers=True)
        if not train and self._val_loader is None:
            dataset = VocosDataset(cfg, train=train)
            self._val_loader   = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True, persistent_workers=True)

    def train_dataloader(self) -> DataLoader:
        self._set_dataloder(self.train_config, train=True)
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        self._set_dataloder(self.val_config,   train=False)
        return self._val_loader


class VocosDataset(Dataset):
    """Gain-matched audio segment dataset."""

    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist: list[str] = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self._cache: list[None|tuple[Tensor, int]] = [None for _ in range(len(self.filelist))]
        self._cache_cuda = cfg.cache_cuda

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> Tensor:
        """
        Args:
            index - Item index
        Returns:
            :: (T,) - Audio segment
        """

        # Load :: -> (Channel, T)
        if self._cache[index] is None:
            # From disk and make cache
            audio_path = self.filelist[index]
            y, sr = torchaudio.load(audio_path)
            if self._cache_cuda:
                y = y.cuda()
            self._cache[index] = y, sr
        y, sr = self._cache[index]
        assert sr == self.sampling_rate

        # Gain randomize/standardize
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])

        # Padding & Clipping
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            y = y[:, : self.num_samples]

        return y[0]
