# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

# Main script for sitw/ivector recipe.

import sys
import os
# Adding the project root to the path to make imports to work regardless from where this file was executed:
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('asvtorch', 1)[0])
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import numpy as np

from asvtorch.src.settings.abstract_settings import AbstractSettings
from asvtorch.src.settings.settings import Settings
import asvtorch.recipes.sitw.data_preparation as data_preparation  # same preparation as in deep embedding recipe
import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.frontend.feature_extractor import FeatureExtractor
from asvtorch.src.utterances.utterance_selector import UtteranceSelector
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.ivector.kaldi_ubm_training import train_kaldi_ubm
from asvtorch.src.backend.vector_processing import VectorProcessor
from asvtorch.src.backend.plda import Plda
from asvtorch.src.evaluation.scoring import score_trials_plda, prepare_scoring
from asvtorch.src.evaluation.eval_metrics import compute_eer, compute_min_dcf
import asvtorch.src.backend.score_normalization as score_normalization
import asvtorch.src.misc.recipeutils as recipeutils
from asvtorch.src.ivector.posteriors import extract_posteriors
from asvtorch.src.ivector.ivector_extractor import IVectorExtractor
from asvtorch.src.ivector.gmm import Gmm

@dataclass
class RecipeSettings(AbstractSettings):
    start_stage: int = 0
    end_stage: int = 100
    preparation_datasets: Optional[List[str]] = None
    feature_extraction_datasets: List[str] = field(default_factory=lambda:[])
    augmentation_datasets: Dict[str, int] = field(default_factory=lambda:{})
    selected_iteration: int = 5

# Initializing settings:
Settings(os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'init_config.py')) 

# Set the configuration file for KALDI MFCCs:
Settings().paths.kaldi_mfcc_conf_file = os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'mfcc.conf')

# Add recipe settings to Settings() (these settings may not be reusable enough to be included in settings.py)
Settings().recipe = RecipeSettings()  

# Get full path of run config file:
run_config_file = os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'run_configs.py') 

# Get run configs from command line arguments
run_configs = sys.argv[1:]
if not run_configs:
    sys.exit('Give one or more run configs as argument(s)!')

Settings().print()

# SITW trial lists:
trial_list_list = [
    recipeutils.TrialList(trial_list_display_name='SITW core-core', dataset_folder='sitw', trial_file='sitw_trials_core_core.txt'),
    recipeutils.TrialList(trial_list_display_name='SITW core-multi', dataset_folder='sitw', trial_file='sitw_trials_core_multi.txt')
]

# Run config loop:
for settings_string in Settings().load_settings(run_config_file, run_configs):

    # Preparation, stage 0
    if Settings().recipe.start_stage <= 0 <= Settings().recipe.end_stage:     
        data_preparation.prepare_datasets(Settings().recipe.preparation_datasets)

    # Feature extraction, stage 1
    if Settings().recipe.start_stage <= 1 <= Settings().recipe.end_stage:
        for dataset in Settings().recipe.feature_extraction_datasets:
            FeatureExtractor().extract_features(dataset)

    # Data augmentation, stage 2
    if Settings().recipe.start_stage <= 2 <= Settings().recipe.end_stage:
        for dataset, augmentation_factor in Settings().recipe.augmentation_datasets.items():
            FeatureExtractor().augment(dataset, augmentation_factor)

    # GMM-UBM training, stage 4
    if Settings().recipe.start_stage <= 3 <= Settings().recipe.end_stage:
        train_kaldi_ubm('voxceleb1')


    # Component posterior extraction, stage 5
    if Settings().recipe.start_stage <= 4 <= Settings().recipe.end_stage:

        trial_data = recipeutils.get_trial_utterance_list(trial_list_list)
        extract_posteriors(trial_data)
        trial_data.save('trial_posteriors', folder=fileutils.get_posterior_folder())
        
        training_data = UtteranceSelector().choose_all('voxceleb1')
        training_data.combine(UtteranceSelector().choose_all('voxceleb2'))
        extract_posteriors(training_data)
        training_data.save('training_posteriors', folder=fileutils.get_posterior_folder())


    # I-vector extractor training, stage 5
    if Settings().recipe.start_stage <= 5 <= Settings().recipe.end_stage:
        training_data = UtteranceList.load('training_posteriors', folder=fileutils.get_posterior_folder())
        training_data.remove_short_utterances(500)  # Remove utts with less than 500 frames
        training_data.choose_random(100000)  # Retain 100000 randomly selected utterances

        ubm = Gmm.from_kaldi(fileutils.get_ubm_file(), Settings().computing.device)
        ivector_extractor = IVectorExtractor.random_init(ubm, Settings().computing.device)
        ivector_extractor.train(training_data)

    
    # I-vector extraction, stage 7
    if Settings().recipe.start_stage <= 7 <= Settings().recipe.end_stage:

        ivector_extractor = IVectorExtractor.from_npz(Settings().recipe.selected_iteration, Settings().computing.device)

        print('Loading trial data...')
        trial_data = UtteranceList.load('trial_posteriors', folder=fileutils.get_posterior_folder())
        ivector_extractor.extract(trial_data)
        trial_data.save('trial_ivectors')

        print('Loading PLDA data...')
        training_data = UtteranceList.load('training_posteriors', folder=fileutils.get_posterior_folder())
        ivector_extractor.extract(training_data)
        training_data.save('plda_ivectors')
        
        ivector_extractor = None


    # I-vector processing, PLDA training, Scoring, Score normalization, stage 9
    if Settings().recipe.start_stage <= 9 <= Settings().recipe.end_stage:

        trial_data = UtteranceList.load('trial_ivectors')
        plda_data = UtteranceList.load('plda_ivectors')

        vector_processor = VectorProcessor.train(plda_data.embeddings, 'cl', Settings().computing.device)
        trial_data.embeddings = vector_processor.process(trial_data.embeddings)
        plda_data.embeddings = vector_processor.process(plda_data.embeddings)
        
        plda = Plda.train_closed_form(plda_data.embeddings, plda_data.get_spk_labels(), Settings().computing.device)

        # Select score normalization cohort randomly from PLDA training data
        # torch.manual_seed(0)
        # normalization_embeddings = plda_data.embeddings[torch.randperm(plda_data.embeddings.size()[0])[:Settings().backend.score_norm_full_cohort_size], :]

        # # Compute s-norm statistics
        # normalization_stats = score_normalization.compute_adaptive_snorm_stats(trial_data.embeddings, normalization_embeddings, plda, Settings().backend.plda_dim, Settings().backend.score_norm_adaptive_cohort_size)

        # Scoring
        for trial_list in trial_list_list:
            trial_file = trial_list.get_path_to_trial_file()
            labels, indices = prepare_scoring(trial_data, trial_file)
            scores = score_trials_plda(trial_data, indices, plda)
            #scores = score_normalization.apply_snorm(scores, normalization_stats, indices)
            np.savetxt(fileutils.get_score_output_file(trial_list), scores)
            eer = compute_eer(scores, labels)[0] * 100
            min_dcf = compute_min_dcf(scores, labels, 0.05, 1, 1)[0]
            output_text = 'EER = {:.4f}  minDCF = {:.4f}  [iteration {}] [{}]'.format(eer, min_dcf, Settings().recipe.selected_iteration, trial_list.trial_list_display_name)
            print(output_text)

print('All done!')
