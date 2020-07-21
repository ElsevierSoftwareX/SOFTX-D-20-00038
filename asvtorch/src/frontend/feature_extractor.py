# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from asvtorch.src.settings.settings import Settings
from asvtorch.src.misc.singleton import Singleton
from asvtorch.src.frontend.kaldi.kaldi_operations import extract_kaldi_features, kaldi_augment

class FeatureExtractor(metaclass=Singleton):
    def __init__(self):
        self.feature_extractor_func = None
        self.augmenting_func = None
        self.refresh_feature_extractor()
        self.refresh_augmenter()

    # Can be useful if multiple kinds of features are used in the same project (updates the settings if they have changed)
    def refresh_feature_extractor(self):
        if Settings().features.use_kaldi:
            self.feature_extractor_func = extract_kaldi_features
        else:
            raise NotImplementedError

    def refresh_augmenter(self):
        if Settings().features.use_kaldi:
            self.augmenting_func = kaldi_augment
        else:
            raise NotImplementedError

    def extract_features(self, dataset: str):
        return self.feature_extractor_func(dataset)

    def augment(self, dataset: str, augmentation_factor: int):
        return self.augmenting_func(dataset, augmentation_factor)
