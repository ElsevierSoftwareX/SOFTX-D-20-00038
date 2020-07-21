# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import numpy as np

import kaldi.feat.functions as featfuncs
import kaldi.util.io as kio

from asvtorch.src.settings.settings import Settings

from asvtorch.src.utterances.utterance import Utterance


class KaldiFeatureLoader():
    def __init__(self):

        deltas = Settings().features.deltas
        self.deltas = False
        if deltas != 'b':
            self.deltas = True
            if 'b' in deltas and 'v' in deltas and 'a' in deltas:
                self.delta_opts = featfuncs.DeltaFeaturesOptions(order=2, window=Settings().features.delta_reach)
            elif 'b' in deltas and 'v' in deltas:
                self.delta_opts = featfuncs.DeltaFeaturesOptions(order=1, window=Settings().features.delta_reach)
            else:
                raise NotImplementedError

        cmvn = Settings().features.cmvn
        self.cmvn = bool(cmvn)
        if self.cmvn:
            self.cmvn_opts = featfuncs.SlidingWindowCmnOptions()
            self.cmvn_opts.center = True
            self.cmvn_opts.normalize_variance = True if 'v' in Settings().features.cmvn else False
            self.cmvn_opts.cmn_window = Settings().features.cmvn_window

    def load_features(self, utterance: Utterance) -> np.ndarray:
        features = []
        for feature_location, frame_selector in utterance.generator():
            feats = kio.read_matrix(feature_location)
            if self.deltas:
                feats = featfuncs.compute_deltas(self.delta_opts, feats)
            if self.cmvn:
                featfuncs.sliding_window_cmn(self.cmvn_opts, feats, feats)
            feats = feats.numpy()
            feats = frame_selector.select(feats, utterance.utt_id)
            features.append(feats)
        return np.vstack(features)
