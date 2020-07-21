# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import numpy as np

from asvtorch.src.settings.settings import Settings
from asvtorch.src.misc.singleton import Singleton
from asvtorch.src.frontend.featureloaders.kaldi_featureloader import KaldiFeatureLoader
from asvtorch.src.utterances.utterance import Utterance

class FeatureLoader(metaclass=Singleton):
    def __init__(self):
        self.refresh_feature_loader()

    # Can be useful if multiple kinds of features are used in the same project (updates the settings if they have changed)
    def refresh_feature_loader(self):
        if Settings().features.use_kaldi:
            self.feature_loader = KaldiFeatureLoader()
        else:
            raise NotImplementedError

    def load_features(self, utterance: Utterance) -> np.ndarray:
        return self.feature_loader.load_features(utterance)

    def get_feature_dimension(self, utterance: Utterance) -> int:
        return self.feature_loader.load_features(utterance).shape[1]
