# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

# Main script for voxceleb/neural_ivector recipe.

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

import asvtorch.recipes.voxceleb.data_preparation as data_preparation

import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.misc.miscutils import dual_print

from asvtorch.src.frontend.feature_extractor import FeatureExtractor
from asvtorch.src.utterances.utterance_selector import UtteranceSelector
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.networks.network_testing import extract_embeddings, extract_stats
import asvtorch.src.networks.network_training as network_training
import asvtorch.src.networks.network_io as network_io
from asvtorch.src.backend.vector_processing import VectorProcessor
from asvtorch.src.backend.plda import Plda
from asvtorch.src.evaluation.scoring import score_trials_plda, prepare_scoring
from asvtorch.src.evaluation.eval_metrics import compute_eer, compute_min_dcf
import asvtorch.src.backend.score_normalization as score_normalization
import asvtorch.src.misc.recipeutils as recipeutils
from asvtorch.src.ivector.ivector_extractor import IVectorExtractor
from asvtorch.src.ivector.gmm import Gmm


@dataclass
class RecipeSettings(AbstractSettings):
    start_stage: int = 0
    end_stage: int = 100
    preparation_datasets: Optional[List[str]] = None
    feature_extraction_datasets: List[str] = field(default_factory=lambda: [])
    augmentation_datasets: Dict[str, int] = field(default_factory=lambda: {})
    selected_epoch: int = 5
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

# These small trial lists are used between epochs to compute EERs:
small_trial_list_list = [
    recipeutils.TrialList(trial_list_display_name='vox1_original', dataset_folder='voxceleb1', trial_file='veri_test.txt'),
    recipeutils.TrialList(trial_list_display_name='vox1_cleaned', dataset_folder='voxceleb1', trial_file='veri_test2.txt')
]

# These are the all VoxCeleb trial lists used for final testing:
full_trial_list_list = [
    *small_trial_list_list,
    recipeutils.TrialList(trial_list_display_name='vox1_extended_original', dataset_folder='voxceleb1', trial_file='list_test_all.txt'),
    recipeutils.TrialList(trial_list_display_name='vox1_extended_cleaned', dataset_folder='voxceleb1', trial_file='list_test_all2.txt'),
    recipeutils.TrialList(trial_list_display_name='vox1_hard_original', dataset_folder='voxceleb1', trial_file='list_test_hard.txt'),
    recipeutils.TrialList(trial_list_display_name='vox1_hard_cleaned', dataset_folder='voxceleb1', trial_file='list_test_hard2.txt')
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

    # Network training, stage 5
    if Settings().recipe.start_stage <= 5 <= Settings().recipe.end_stage:

        print('Selecting network training data...')
        training_data = UtteranceSelector().choose_all('voxceleb2_cat_combined') # combined = augmented version
        #training_data = UtteranceSelector().choose_all('voxceleb2_cat') # non-augmented
        training_data.remove_short_utterances(500)  # Remove utts with less than 500 frames
        training_data.remove_speakers_with_few_utterances(10)  # Remove spks with less than 10 utts

        print('Selecting PLDA training data...')
        plda_data = UtteranceSelector().choose_all('voxceleb2_cat_combined')
        #plda_data = UtteranceSelector().choose_random('voxceleb2_cat', 40000) # non-augmented
        plda_data.select_random_speakers(500)

        trial_data = recipeutils.get_trial_utterance_list(small_trial_list_list)

        result_file = open(fileutils.get_new_results_file(), 'w')
        result_file.write(settings_string + '\n\n')

        eer_stopper = recipeutils.EerStopper()

        for epoch in range(Settings().network.resume_epoch, Settings().network.max_epochs, Settings().network.epochs_per_train_call):

            network, stop_flag, epoch = network_training.train_network(training_data, epoch)

            extract_embeddings(trial_data, network)
            extract_embeddings(plda_data, network)
            network = None

            vector_processor = VectorProcessor.train(plda_data.embeddings, 'cwl', Settings().computing.device)
            trial_data.embeddings = vector_processor.process(trial_data.embeddings)
            plda_data.embeddings = vector_processor.process(plda_data.embeddings)

            plda = Plda.train_closed_form(plda_data.embeddings, plda_data.get_spk_labels(), Settings().computing.device)

            for trial_list in small_trial_list_list:
                trial_file = trial_list.get_path_to_trial_file()
                labels, indices = prepare_scoring(trial_data, trial_file)
                scores = score_trials_plda(trial_data, indices, plda)
                eer = compute_eer(scores, labels)[0] * 100
                eer_stopper.add_stopping_eer(eer)
                min_dcf = compute_min_dcf(scores, labels, 0.05, 1, 1)[0]
                output_text = 'EER = {:.4f}  minDCF = {:.4f}  [epoch {}] [{}]'.format(eer, min_dcf, epoch, trial_list.trial_list_display_name)
                dual_print(result_file, output_text)
            dual_print(result_file, '')

            trial_data.embeddings = None  # Release GPU memory (?)
            plda_data.embeddings = None
            torch.cuda.empty_cache()

            if eer_stopper.stop() or stop_flag:
                break

        result_file.close()


    # Embedding extraction, stage 7
    if Settings().recipe.start_stage <= 7 <= Settings().recipe.end_stage:
        epoch = Settings().recipe.selected_epoch if Settings().recipe.selected_epoch else recipeutils.find_last_epoch()
        network = network_io.load_network(epoch, Settings().computing.device)

        print('Loading trial data...')
        trial_data = recipeutils.get_trial_utterance_list(full_trial_list_list)
        print('Loading PLDA data...')
        plda_data = UtteranceSelector().choose_all('voxceleb2_cat_combined') # use the whole data in testing mode
        #plda_data = UtteranceSelector().choose_all('voxceleb2_cat') # non-augmented

        print('Extracting trial embeddings...')
        extract_embeddings(trial_data, network)
        trial_data.save('trial_embeddings')

        print('Extracting PLDA embeddings...')
        extract_embeddings(plda_data, network)
        plda_data.save('plda_embeddings')

        network = None


    # Embedding processing, PLDA training, Scoring, Score normalization, stage 9
    if Settings().recipe.start_stage <= 9 <= Settings().recipe.end_stage:
        epoch = Settings().recipe.selected_epoch if Settings().recipe.selected_epoch else recipeutils.find_last_epoch()

        trial_data = UtteranceList.load('trial_embeddings')
        plda_data = UtteranceList.load('plda_embeddings')

        vector_processor = VectorProcessor.train(plda_data.embeddings, 'cwl', Settings().computing.device)
        trial_data.embeddings = vector_processor.process(trial_data.embeddings)
        plda_data.embeddings = vector_processor.process(plda_data.embeddings)

        plda = Plda.train_closed_form(plda_data.embeddings, plda_data.get_spk_labels(), Settings().computing.device)

        # Select score normalization cohort randomly from PLDA training data
        #torch.manual_seed(0)
        #normalization_embeddings = plda_data.embeddings[torch.randperm(plda_data.embeddings.size()[0])[:Settings().backend.score_norm_full_cohort_size], :]

        # Compute s-norm statistics
        #normalization_stats = score_normalization.compute_adaptive_snorm_stats(trial_data.embeddings, normalization_embeddings, plda, Settings().backend.plda_dim, Settings().backend.score_norm_adaptive_cohort_size)

        # Scoring and score normalization
        for trial_list in full_trial_list_list:
            trial_file = trial_list.get_path_to_trial_file()
            labels, indices = prepare_scoring(trial_data, trial_file)
            scores = score_trials_plda(trial_data, indices, plda)  
            #scores = score_normalization.apply_snorm(scores, normalization_stats, indices)
            np.savetxt(fileutils.get_score_output_file(trial_list, 'dnn_embedding'), scores)
            eer = compute_eer(scores, labels)[0] * 100
            min_dcf = compute_min_dcf(scores, labels, 0.05, 1, 1)[0]
            output_text = 'EER = {:.4f}  minDCF = {:.4f}  [epoch {}] [{}]'.format(eer, min_dcf, epoch, trial_list.trial_list_display_name)
            print(output_text)


    # Extracting sufficient statistics, stage 11
    if Settings().recipe.start_stage <= 11 <= Settings().recipe.end_stage:
        epoch = Settings().recipe.selected_epoch if Settings().recipe.selected_epoch else recipeutils.find_last_epoch()
        network = network_io.load_network(epoch, Settings().computing.device)

        print('Selecting i-vector extractor and PLDA training data...')
        training_data = UtteranceSelector().choose_all('voxceleb2_cat_combined') # combined = augmented version
        #training_data = UtteranceSelector().choose_all('voxceleb2_cat') # non-augmented
        training_data.remove_short_utterances(500)  # Remove utts with less than 500 frames
        extract_stats(training_data, network, second_order=True)
        training_data.save('ivector_training_stats')

        trial_data = recipeutils.get_trial_utterance_list(full_trial_list_list)
        extract_stats(trial_data, network, second_order=False)
        trial_data.save('ivector_trial_stats')


    # I-vector extractor training, stage 12
    if Settings().recipe.start_stage <= 12 <= Settings().recipe.end_stage:
        training_data = UtteranceList.load('ivector_training_stats')

        ubm = Gmm.from_stats(training_data)
        ivector_extractor = IVectorExtractor.random_init(ubm, Settings().computing.device)
        ivector_extractor.train(training_data)


    # I-vector extraction, stage 13
    if Settings().recipe.start_stage <= 13 <= Settings().recipe.end_stage:

        ivector_extractor = IVectorExtractor.from_npz(Settings().recipe.selected_iteration, Settings().computing.device)

        print('Loading trial data...')
        trial_data = UtteranceList.load('ivector_trial_stats')
        ivector_extractor.extract(trial_data)
        delattr(trial_data, 'stats') # No need to save stats again
        trial_data.save('trial_ivectors')

        print('Loading PLDA data...')
        training_data = UtteranceList.load('ivector_training_stats')
        ivector_extractor.extract(training_data)
        delattr(training_data, 'stats')
        training_data.save('plda_ivectors')

        ivector_extractor = None


    # I-vector processing, PLDA training, Scoring, Score normalization, stage 15
    if Settings().recipe.start_stage <= 15 <= Settings().recipe.end_stage:

        trial_data = UtteranceList.load('trial_ivectors')
        plda_data = UtteranceList.load('plda_ivectors')

        vector_processor = VectorProcessor.train(plda_data.embeddings, 'cwl', Settings().computing.device)
        trial_data.embeddings = vector_processor.process(trial_data.embeddings)
        plda_data.embeddings = vector_processor.process(plda_data.embeddings)

        plda = Plda.train_closed_form(plda_data.embeddings, plda_data.get_spk_labels(), Settings().computing.device)

        # Select score normalization cohort randomly from PLDA training data
        #torch.manual_seed(0)
        #normalization_embeddings = plda_data.embeddings[torch.randperm(plda_data.embeddings.size()[0])[:Settings().backend.score_norm_full_cohort_size], :]

        # Compute s-norm statistics
        #normalization_stats = score_normalization.compute_adaptive_snorm_stats(trial_data.embeddings, normalization_embeddings, plda, Settings().backend.plda_dim, Settings().backend.score_norm_adaptive_cohort_size)

        # Scoring and score normalization
        for trial_list in full_trial_list_list:
            trial_file = trial_list.get_path_to_trial_file()
            labels, indices = prepare_scoring(trial_data, trial_file)
            scores = score_trials_plda(trial_data, indices, plda)  
            #scores = score_normalization.apply_snorm(scores, normalization_stats, indices)
            np.savetxt(fileutils.get_score_output_file(trial_list, 'neural_ivector'), scores)
            eer = compute_eer(scores, labels)[0] * 100
            min_dcf = compute_min_dcf(scores, labels, 0.05, 1, 1)[0]
            output_text = 'EER = {:.4f}  minDCF = {:.4f}  [iteration {}] [{}]'.format(eer, min_dcf, Settings().recipe.selected_iteration, trial_list.trial_list_display_name)
            print(output_text)

print('All done!')
            