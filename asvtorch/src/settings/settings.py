# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import sys
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

import torch
import torch.cuda

from asvtorch.src.settings.abstract_settings import AbstractSettings

# Edit configs using separate (recipe specific) settings files unless you want to the change the default values permanently.

@dataclass
class PathSettings(AbstractSettings):
    kaldi_recipe_folder: str = '/path/to/your/kaldi/egs/xxx/vx/'
    kaldi_mfcc_conf_file: str = 'path/to/mfcc/conf/file'
    musan_folder: str = '/path/to/musan/'  # Used in Kaldi's augmentation
    datasets: Dict[str, str] = field(default_factory=lambda: {'example_dataset_name': '/example/input_dataset/folder/'})
    output_folder: str = '/all/the/system/outputs/go/here/'
    feature_and_list_folder: str = 'kaldi'  # Relative folder for acoustic features and lists
    system_folder: str = 'system1'  # Relative folder for system (contains a network, xvectors, output scores, etc...)

@dataclass
class ComputingSettings(AbstractSettings):
    ubm_training_workers: int = 8
    posterior_extraction_dataloader_workers: int = 5
    ivector_dataloader_workers: int = 10
    network_dataloader_workers: int = 10
    feature_extraction_workers: int = 44
    use_gpu: bool = True  # If false, then use CPU
    gpu_ids: Tuple[int] = (0,)  # Which GPUs to use?
    
    local_process_rank: int = 0  # Automatically updated by distributed computing scripts
    local_gpu_id: int = 0  # Automatically updated by distributed computing scripts
    world_size: int = 1  # How many processes in distributed computing? Automatically set.
    device: torch.device = field(init=False)  # Automatically set.


@dataclass
class NetworkSettings(AbstractSettings):
    resume_epoch: int = 0  # Continue training from the given epoch (0 = start training new model)
    max_epochs: int = 1000  # Stop training after this number of epochs (if not already stopped by other means)
    eer_stop_epochs: int = 3  # Stop training if EER does not improve in this many epochs
    minimum_improvement: float = 0.1  # What change in EER is considered as an improvement (related to previous stopping criterion)
    epochs_per_train_call: int = 1  # Increase this if you do not want to compute EER after every epoch

    utts_per_speaker_in_epoch: int = 320  # How large is one "epoch"?

    minibatch_size: int = 64  # (number of utterances)

    min_clip_size: int = 200  # Minimum training utterance duration in frames
    max_clip_size: int = 400

    max_test_cut_portion: float = 0.02  # Maximum portion of utterance to be cutted out in embedding extraction
    max_val_cut_portion: float = 0.1   # Maximum portion of utterance to be cutted out in network validation
    max_batch_size_in_frames: int = 30000  # Max batch size in frames in embedding extraction and validation

    validation_utterances: int = 500  # Number of utterances used for validation

    print_interval: int = 500  # How often to print traninig loss
    accuracy_print_interval: int = 6  # Evaluate accuracy every nth print (Too small value will affect speed)
    extraction_print_interval: int = 200  # Print interval in embedding extraction (in batches)

    optimizer: str = 'sgd'

    max_consecutive_lr_updates = 2
    initial_learning_rate: float = 0.1
    min_loss_change_ratio: float = 0.01
    min_room_for_improvement: float = 0.1
    target_loss: float = 0.1

    momentum: float = 0  # (0 to disable)

    weight_decay: float = 0.001    # Weight decay for utterance-level layers
    weight_decay_skiplist: Tuple[str] = ('batchnorm',)

    optimizer_step_interval: int = 1  # This can be used to combine gradients of many minibatches before updating weights

    cnn_padding_mode: str = 'zeros'  # or 'circular'

    # Batch normalization settings:
    bn_momentum: int = 0.1  # (PyTorch default value = 0.1)
    bn_affine: bool = True  # (PyTorch default value = True)

    # Activation settings:
    activation: str = 'lrelu' # relu/lrelu/selu
    lrelu_slope: float = 0.01 # Used when lrelu is selected

    # Network architecture:
    network_class: str = 'asvtorch.src.networks.architectures.StandardNet'
    frame_layer_size: int = 512
    stat_size: int = 1500
    embedding_size: int = 512
    
    pooling_layer_type: str = 'default'  # or 'clustering'

    cluster_assignment_mode: str = 'net_vlad'  # or 'lde'
    supervector_mode: str = 'net_vlad'  # or 'lde'
    lde_covariance_type: str = 'spherical'  # full/diagonal/spherical/shared_full/shared_diagonal/shared_spherical
    normalize_supervector: bool = True  # Length norm of supervector
    n_clusters: int = 64  # used in pooling layers that have components
    n_ghost_clusters: int = 2
    net_vlad_alpha: float = 1

    ser: int = 16  # parameter for squeeze-and-excite
    attention_layer_size: int = 64  # parameter for attention module


@dataclass
class IVectorSettings(AbstractSettings):
    # general settings
    num_gaussians: int = 2048
    ubm_name: str = 'ubm_2048'
    ivec_dim: int = 400
    ivec_type: str = 'augmented' # or 'standard'

    # training settings
    n_iterations: int = 5
    initial_prior_offset: float = 100  # Only used in the augmented formulation
    covariance_type: str = 'full'  # full/diagonal/spherical/shared_full/shared_diagonal/shared_spherical
    update_projections: bool = True
    update_covariances: bool = True
    minimum_divergence: bool = True
    update_means: bool = True

    # data loading & batching settings
    batch_size_in_utts: int = 200   # Higher batch size will have higher GPU memory usage.
    n_component_batches: int = 16   # must be a power of two! The higher the value, the less GPU memory will be used.


@dataclass
class PosteriorExtractionSettings(AbstractSettings):
    n_top_gaussians: int = 20  # How many top (diagonal) Gaussians to select for full covariance GMM posterior computation
    posterior_threshold: float = 0.025  # posteriors with lower than this value will be neglegted

    # data loading & batching settings
    batch_size_in_frames: int = 500000



@dataclass
class FeatureSettings(AbstractSettings):
    # Not all settings are used with kaldi features; some settings are for standalone features that are not currently implemented:
    sampling_rate: int = 16000
    frame_shift: int = 10  # in milliseconds
    use_kaldi: bool = True  # Use Kaldi for feature extraction
    cmvn: str = 'm'  # m = mean normalization, mv = mean & variance normalization, or empty for no cmvn
    cmvn_window: int = 300  # Sliding window size in frames
    deltas: str = 'b'  # b = base coefficients, v = deltas (velocity), a = double deltas (acceleration) (for example 'va' to have all but the base coeffs)
    delta_reach: int = 3  # delta window reach in one direction
    sad_enabled: bool = True  # speech activity detection
    log_feats: bool = True  # Apply log
    dct_feats: bool = True  # Apply DCT
    vad_mismatch_tolerance: int = 0  # maximum difference between number of vad labels and number of features

@dataclass
class BackendSettings(AbstractSettings):
    plda_dim: int = 200
    max_all_vs_all_score_count: int = int(100e6)  # if more scores, use pairwise scoring
    pairwise_scoring_chunk_size: int = int(0.05e6)
    score_norm_full_cohort_size: int = 2000
    score_norm_adaptive_cohort_size: int = 200

@dataclass
class DiarizationSettings(AbstractSettings):
    pass

@dataclass
class Settings(AbstractSettings):  
    init_settings_file: str = None
    paths: PathSettings = field(default_factory=lambda: PathSettings(), init=False)
    computing: ComputingSettings = field(default_factory=lambda: ComputingSettings(), init=False)
    network: NetworkSettings = field(default_factory=lambda: NetworkSettings(), init=False)
    posterior_extraction: PosteriorExtractionSettings = field(default_factory=lambda: PosteriorExtractionSettings(), init=False)
    ivector: IVectorSettings = field(default_factory=lambda: IVectorSettings(), init=False)
    features: FeatureSettings = field(default_factory=lambda: FeatureSettings(), init=False)
    backend: BackendSettings = field(default_factory=lambda: BackendSettings(), init=False)
    diarization: DiarizationSettings = field(default_factory=lambda: DiarizationSettings(), init=False)

    def __post_init__(self):
        # Initial settings
        if self.init_settings_file is not None:
            self.set_initial_settings(self.init_settings_file)

    def post_update_call(self):

        # Detect sampling rate from Kaldi MFCC conf file.
        if self.features.use_kaldi:
            config_file = self.paths.kaldi_mfcc_conf_file
            sampling_rate = -1
            with open(config_file) as f:
                for line in f:
                    if '--sample-frequency' in line:
                        sampling_rate = int(line.split('=')[1].strip())
                        break
            if sampling_rate == -1:
                sys.exit('ERROR: "--sample-frequency" could not be found from Kaldi MFCC config file!')
            else:
                self.features.sampling_rate = sampling_rate
                print('Sampling rate set to {} using Kaldi MFCC config file!'.format(sampling_rate))

        # Set GPU device
        self.computing.device = torch.device("cpu")
        if torch.cuda.is_available():
            self._set_local_gpu_id_using_local_rank()
            self.computing.device = torch.device('cuda:{}'.format(self.computing.local_gpu_id))
            torch.cuda.set_device(self.computing.device)
            torch.backends.cudnn.benchmark = False
            print('Using GPU (gpu_id = {})!'.format(self.computing.local_gpu_id))
        else:
            print('Cuda is not available! Using CPU!')

    def _set_local_gpu_id_using_local_rank(self):
        if len(self.computing.gpu_ids) < self.computing.world_size: # not enough gpu_ids, so using identity mapping
            self.computing.local_gpu_id = self.computing.local_process_rank
        else:
            self.computing.local_gpu_id = self.computing.gpu_ids[self.computing.local_process_rank]
