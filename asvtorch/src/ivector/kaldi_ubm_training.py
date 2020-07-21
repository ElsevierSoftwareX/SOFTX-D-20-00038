# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
import subprocess
from shutil import copyfile

import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.settings.settings import Settings

def train_kaldi_ubm(dataset: str):   
    diag_training_script = os.path.join(Settings().paths.kaldi_recipe_folder, 'sid', 'asvtorch_train_diag_ubm.sh')
    #if not os.path.exists(diag_training_script):
    copyfile(os.path.join(fileutils.get_folder_of_file(__file__), 'asvtorch_train_diag_ubm.sh'), diag_training_script)
    os.chmod(diag_training_script, 0o755)

    full_training_script = os.path.join(Settings().paths.kaldi_recipe_folder, 'sid', 'asvtorch_train_full_ubm.sh')
    #if not os.path.exists(full_training_script):
    copyfile(os.path.join(fileutils.get_folder_of_file(__file__), 'asvtorch_train_full_ubm.sh'), full_training_script)
    os.chmod(full_training_script, 0o755)

    list_folder = fileutils.get_list_folder(dataset)
    num_gauss = str(Settings().ivector.num_gaussians)
    ubm_folder_diag = os.path.join(fileutils.get_ubm_folder(), 'diag_' + Settings().ivector.ubm_name)
    ubm_folder_full = os.path.join(fileutils.get_ubm_folder(), 'full_' + Settings().ivector.ubm_name)
    njobs = str(Settings().computing.ubm_training_workers)
    delta_window = str(Settings().features.delta_reach)

    deltas = Settings().features.deltas
    if 'b' in deltas and 'v' in deltas and 'a' in deltas:
        delta_order = str(2)
    elif 'b' in deltas and 'v' in deltas:
        delta_order = str(1)
    elif 'b' in deltas and 'a' not in deltas:
        delta_order = str(0)
    else:
        raise NotImplementedError

    cmvn = Settings().features.cmvn
    apply_cmn = 'true' if cmvn else 'false'
    norm_vars = 'true' if 'v' in Settings().features.cmvn else 'false'
    cmn_window = str(Settings().features.cmvn_window)

    subprocess.run(['./sid/asvtorch_train_diag_ubm.sh', list_folder, num_gauss, ubm_folder_diag, njobs, delta_window, delta_order, apply_cmn, norm_vars, cmn_window], cwd=Settings().paths.kaldi_recipe_folder)

    subprocess.run(['./sid/asvtorch_train_full_ubm.sh', list_folder, ubm_folder_diag, ubm_folder_full, njobs, apply_cmn, norm_vars, cmn_window], cwd=Settings().paths.kaldi_recipe_folder)
