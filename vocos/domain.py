"""Domain."""

from torch import Tensor


UnitSeries = Tensor # :: (B, Feat, Frame)
FeatSeries = Tensor # :: (B, Frame, Feat)
Wave       = Tensor # :: (B, T)