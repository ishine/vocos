from __future__ import annotations
from typing import Tuple, Any, Union, Dict

import torch
import yaml
from huggingface_hub import hf_hub_download
from torch import nn, Tensor
from vocos.feature_extractors import FeatureExtractor, EncodecFeatures
from vocos.heads import FourierHead
from vocos.models import Backbone
from vocos.domain import UnitSeries, Wave


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class Vocos(nn.Module):
    """Vocos inference model.
    Model load, Reconstruction (wave-to-unit-to-wave), Vocoding (unit-to-wave) and Unitnize(code-to-unit) are supported.
    """

    def __init__(self, feature_extractor: FeatureExtractor, backbone: Backbone, head: FourierHead):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config_path: str) -> Vocos:
        """Create a instance from hyperparameters stored in a yaml configuration file (states are not restored).
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(args=(), init=config["feature_extractor"])
        backbone          = instantiate_class(args=(), init=config["backbone"])
        head              = instantiate_class(args=(), init=config["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(self, repo_id: str) -> Vocos:
        """Create a instance from a pre-trained model stored in the Hugging Face model hub.
        """
        # Instantiate
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
        model = self.from_hparams(config_path)

        # Restore states
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)

        # Switch mode
        model.eval()

        return model

    @torch.inference_mode()
    def forward(self, audio_input: Tensor, **kwargs: Any) -> Wave:
        """Run a copy-synthesis from audio waveform.

        The feature extractor first processes the audio input, which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input :: (B, T) - Input audio waveform
        Returns:
                        :: (B, T) - Reconstructed audio
        """
        return self.decode(self.feature_extractor(audio_input, **kwargs), **kwargs)

    @torch.inference_mode()
    def decode(self, features_input: UnitSeries, **kwargs: Any) -> Wave:
        """Decode audio waveform from acoustic features, backbone + head.

        Args:
            features_input :: (B, Feat, Frame) - Feature series
        Returns:
                           :: (B, T)           - Reconstructed audio
        """
        return self.head(self.backbone(features_input, **kwargs))

    @torch.inference_mode()
    def codes_to_features(self, codes: Tensor) -> Tensor:
        """
        Transforms an input sequence of discrete tokens (codes) into feature embeddings using the feature extractor's codebook weights.

        Args:
            codes :: (K, Frame) | (K, B, Frame) - Code series, K is the number of codebooks

        Returns:
                  :: (B, Feat, Frame)           - Feature series
        """
        assert isinstance(self.feature_extractor, EncodecFeatures), "Feature extractor should be an instance of EncodecFeatures"

        # Reshape :: (K, Frame) -> (K, B=1, Frame)
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)

        n_bins = self.feature_extractor.encodec.quantizer.bins
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.feature_extractor.codebook_weights).sum(dim=0)
        features = features.transpose(1, 2)

        return features
