# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
from os.path import isfile, join
import datetime
from typing import List

from asvtorch.src.misc.recipeutils import TrialList
from asvtorch.src.settings.settings import Settings

def ensure_exists(folder: str):
    """If the folder does not exist, create it.

    Arguments:
        folder {string} -- Folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def ensure_ext(filename: str, ext: str) -> str:
    if not ext.startswith('.'):
        ext = '.' + ext
    if not filename.endswith(ext):
        filename = filename + ext
    return filename

def remove_ext(filename: str, ext: str) -> str:
    if not ext.startswith('.'):
        ext = '.' + ext
    if filename.endswith(ext):
        filename = filename[:-len(ext)]
    return filename

def list_files(folder: str) -> List[str]:
    return [f for f in os.listdir(folder) if isfile(join(folder, f))]


def get_dataset_folder(dataset: str) -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.feature_and_list_folder, dataset)
    ensure_exists(folder)
    return folder

def get_list_folder(dataset: str) -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.feature_and_list_folder, dataset, 'lists')
    ensure_exists(folder)
    return folder

def get_feature_folder(dataset: str) -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.feature_and_list_folder, dataset, 'features')
    ensure_exists(folder)
    return folder

def get_network_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'networks')
    ensure_exists(folder)
    return folder

def get_network_log_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'networks', 'logs')
    ensure_exists(folder)
    return folder

def get_ivector_extractor_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'ivector_extractors')
    ensure_exists(folder)
    return folder

def get_ubm_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, 'ubms')
    ensure_exists(folder)
    return folder

def get_posterior_folder() -> str:
    folder = os.path.join(get_ubm_folder(), 'full_' + Settings().ivector.ubm_name, 'posteriors')
    ensure_exists(folder)
    return folder

def get_ubm_file() -> str:
    folder = os.path.join(get_ubm_folder(), 'full_' + Settings().ivector.ubm_name, 'final.ubm')
    ensure_exists(folder)
    return folder

def get_embedding_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'embeddings')
    ensure_exists(folder)
    return folder

def get_utterance_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'utterances')
    ensure_exists(folder)
    return folder

def get_score_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'scores')
    ensure_exists(folder)
    return folder

def get_score_output_file(trial_list: TrialList, prefix: str = None) -> str:
    if prefix:
        os.path.join(get_score_folder(), 'scores_{}_{}'.format(prefix, trial_list.trial_file))
    return os.path.join(get_score_folder(), 'scores_{}'.format(trial_list.trial_file))

def get_results_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'results')
    ensure_exists(folder)
    return folder

def get_file(dataset: str, trial_file: str) -> str:
    list_folder = get_list_folder(dataset)
    return os.path.join(list_folder, trial_file)

def get_new_results_file() -> str:
    results_folder = get_results_folder()
    return os.path.join(results_folder, 'results_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

def get_folder_of_file(filename: str) -> str:
    return os.path.dirname(os.path.abspath(filename))
