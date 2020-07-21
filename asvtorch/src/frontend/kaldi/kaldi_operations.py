# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
import subprocess
from shutil import copyfile

import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.settings.settings import Settings

def extract_kaldi_features(dataset: str):
    extraction_script = os.path.join(Settings().paths.kaldi_recipe_folder, 'asvtorch_extract_feats.sh')
    if not os.path.exists(extraction_script):
        copyfile(os.path.join(fileutils.get_folder_of_file(__file__), 'asvtorch_extract_feats.sh'), extraction_script)
        os.chmod(extraction_script, 0o755)
    list_folder =  fileutils.get_list_folder(dataset)
    feature_folder = fileutils.get_feature_folder(dataset)
    log_folder = os.path.join(feature_folder, 'logs')
    subprocess.run(['./asvtorch_extract_feats.sh', list_folder, log_folder, feature_folder, Settings().paths.kaldi_mfcc_conf_file, str(Settings().computing.feature_extraction_workers)], cwd=Settings().paths.kaldi_recipe_folder)

def kaldi_augment(dataset: str, aug_factor: int):
    if not os.path.exists(os.path.join(Settings().paths.output_folder, Settings().paths.feature_and_list_folder, dataset, 'musan')):
        make_script = os.path.join(Settings().paths.kaldi_recipe_folder, 'local', 'asvtorch_make_musan.sh')
        copyfile(os.path.join(fileutils.get_folder_of_file(__file__), 'asvtorch_make_musan.sh'), make_script)
        os.chmod(make_script, 0o755)
        subprocess.run(['local/asvtorch_make_musan.sh', Settings().paths.musan_folder, fileutils.get_dataset_folder('musan'), str(Settings().features.sampling_rate)], cwd=Settings().paths.kaldi_recipe_folder)
        musan_script = os.path.join(Settings().paths.kaldi_recipe_folder, 'asvtorch_rirs.sh')
        copyfile(os.path.join(fileutils.get_folder_of_file(__file__), 'asvtorch_rirs.sh'), musan_script)
        os.chmod(musan_script, 0o755)
        subprocess.run(['./asvtorch_rirs.sh'], cwd=Settings().paths.kaldi_recipe_folder)
    augmentation_script = os.path.join(Settings().paths.kaldi_recipe_folder, 'asvtorch_augment.sh')
    copyfile(os.path.join(fileutils.get_folder_of_file(__file__), 'asvtorch_augment.sh'), augmentation_script)
    os.chmod(augmentation_script, 0o755)
    list_folder =  fileutils.get_list_folder(dataset)
    feature_folder = fileutils.get_feature_folder(dataset+'_aug_subset')
    log_folder = os.path.join(feature_folder, 'logs')

    with open(os.path.join(list_folder, 'utt2spk')) as f:
        for n_utts, _ in enumerate(f):
            pass
    n_utts += 1
    n_aug_utts = int(n_utts * (aug_factor-1))

    list_folder = fileutils.get_dataset_folder(dataset)

    print('{} contains {} utterances; augmenting {} new utterances...'.format(dataset, n_utts, n_aug_utts))

    subprocess.run(['./asvtorch_augment.sh', list_folder, log_folder, feature_folder, Settings().paths.kaldi_mfcc_conf_file, str(Settings().computing.feature_extraction_workers), str(Settings().features.sampling_rate), str(n_aug_utts)], cwd=Settings().paths.kaldi_recipe_folder)
