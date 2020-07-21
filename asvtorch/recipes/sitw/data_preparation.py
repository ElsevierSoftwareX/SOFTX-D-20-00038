# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import List, Optional
import os
import subprocess

from asvtorch.src.misc.fileutils import ensure_exists
from asvtorch.src.settings.settings import Settings
import asvtorch.src.misc.fileutils as fileutils


def prepare_datasets(datasets: Optional[List[str]]):
    dataset2func = {
        'sitw': prepare_sitw,
        'voxceleb1': prepare_voxceleb1,
        'voxceleb2': prepare_voxceleb2
    }
    if datasets is None:  # If no list is given, prepare all
        datasets = dataset2func.keys()
    for dataset in datasets:
        dataset2func[dataset]()
        print(sep='\n\n')


def prepare_sitw():
    print('Preparing SITW...')
    sitw_folder = Settings().paths.datasets['sitw']
    output_folder = fileutils.get_list_folder('sitw')
    ensure_exists(output_folder)
    enroll_file = os.path.join(sitw_folder, 'eval', 'lists', 'enroll-core.lst')
    core_trials_file = os.path.join(sitw_folder, 'eval', 'keys', 'core-core.lst')
    multi_trials_file = os.path.join(sitw_folder, 'eval', 'keys', 'core-multi.lst')
    output_core_trials_file = os.path.join(output_folder, 'sitw_trials_core_core.txt')
    output_multi_trials_file = os.path.join(output_folder, 'sitw_trials_core_multi.txt')
    files = set()
    enroll2utt = {}
    with open(enroll_file) as f:
        for line in f:
            parts = line.split()
            utt_id = parts[1].split('/')[1].strip()
            files.add(utt_id)
            enroll2utt[parts[0]] = utt_id
    for filenames in [(core_trials_file, output_core_trials_file), (multi_trials_file, output_multi_trials_file)]:
        with open(filenames[0]) as f, open(filenames[1], 'w') as outf:
            for line in f:
                parts = line.split()
                utt_id = parts[1].split('/')[1]
                files.add(utt_id)
                label = parts[2].strip()
                label = 'nontarget' if label == 'imp' else 'target'                   
                outf.write('{} {} {}\n'.format(enroll2utt[parts[0]], utt_id, label))
    wav_output = os.path.join(output_folder, 'wav.scp')
    utt2spk_output = os.path.join(output_folder, 'utt2spk')
    with open(wav_output, 'w') as wav_out, open(utt2spk_output, 'w') as utt2spk_out:
        for filename in files:
            filepath = os.path.join(sitw_folder, 'eval', 'audio', filename)
            wav_out.write('{} ffmpeg -y -v 8 -i {} -ac 1 -ar {} -f wav - |\n'.format(filename, filepath, Settings().features.sampling_rate))
            utt2spk_out.write('{0} {0}\n'.format(filename)) 
    finalize(output_folder)
    print('SITW prepared!')

def _travel_voxceleb_folders(data_folders):
    file_dict = {}
    for data_folder in data_folders:
        spk_folders = os.listdir(data_folder)
        for spk in spk_folders:
            spk_folder = os.path.join(data_folder, spk)
            video_folders = os.listdir(spk_folder)
            for video in video_folders:
                video_folder = os.path.join(spk_folder, video)
                files = os.listdir(video_folder)
                files = [os.path.join(video_folder, filename) for filename in files]
                file_dict[(spk, video)] = files
    return file_dict

def _prepare_voxceleb(data_folder, output_folder, output_folder_cat, wav_func, cat_wav_func):
    file_dict = _travel_voxceleb_folders(data_folder)
    ensure_exists(output_folder)
    wav_output = os.path.join(output_folder, 'wav.scp')
    utt2spk_output = os.path.join(output_folder, 'utt2spk')
    ensure_exists(os.path.join(output_folder_cat, 'cat'))
    wav_output_cat = os.path.join(output_folder_cat, 'wav.scp')
    utt2spk_output_cat = os.path.join(output_folder_cat, 'utt2spk')
    with open(wav_output, 'w') as wav_out, open(utt2spk_output, 'w') as utt2spk_out, open(wav_output_cat, 'w') as wav_out_cat, open(utt2spk_output_cat, 'w') as utt2spk_out_cat:
        for key in file_dict:
            cat_file = os.path.join(output_folder_cat, 'cat', '{}-{}.txt'.format(key[0], key[1]))
            with open(cat_file, 'w') as cat_out:
                for filename in file_dict[key]:
                    utt_id = filename.split('/')[-1].split('.')[0]
                    utt_id = '-'.join((key[0], key[1], utt_id))
                    wav_out.write('{} {}\n'.format(utt_id, wav_func(filename)))
                    utt2spk_out.write('{} {}\n'.format(utt_id, key[0]))
                    cat_out.write("file '{}'\n".format(filename))
            utt_id = '-'.join((key[0], key[1]))
            wav_out_cat.write('{} {}\n'.format(utt_id, cat_wav_func(key)))
            utt2spk_out_cat.write('{} {}\n'.format(utt_id, key[0]))
    finalize(output_folder)
    finalize(output_folder_cat)
    
def prepare_voxceleb1():
    print('Preparing VoxCeleb1...')
    data_folders = (os.path.join(Settings().paths.datasets['voxceleb1'], 'wav'), )
    output_folder = fileutils.get_list_folder('voxceleb1')
    output_folder_cat = fileutils.get_list_folder('voxceleb1_cat')
    def wav_func(filename):
        return filename
    def cat_wav_func(key):
        cat_file = os.path.join(output_folder_cat, 'cat', '{}-{}.txt'.format(key[0], key[1]))
        return 'ffmpeg -f concat -safe 0 -i {} -c copy -f wav - |'.format(cat_file)
    _prepare_voxceleb(data_folders, output_folder, output_folder_cat, wav_func, cat_wav_func)
    print('VoxCeleb1 prepared!')

def prepare_voxceleb2():
    print('Preparing VoxCeleb2...')
    output_folder = fileutils.get_list_folder('voxceleb2')
    output_folder_cat = fileutils.get_list_folder('voxceleb2_cat')
    vox2_folder = Settings().paths.datasets['voxceleb2']
    data_folders = (os.path.join(vox2_folder, 'dev', 'aac'), os.path.join(vox2_folder, 'test', 'aac'))
    def wav_func(filename):
        return 'ffmpeg -y -v 8 -i {} -ac 1 -ar {} -f wav - |'.format(filename, Settings().features.sampling_rate)
    def cat_wav_func(key):
        cat_file = os.path.join(output_folder_cat, 'cat', '{}-{}.txt'.format(key[0], key[1]))
        return 'ffmpeg -f concat -safe 0 -i {} -c copy -f adts pipe: | ffmpeg -y -v 8 -i pipe: -ac 1 -ar {} -f wav - |'.format(cat_file, Settings().features.sampling_rate)
    _prepare_voxceleb(data_folders, output_folder, output_folder_cat, wav_func, cat_wav_func)
    print('VoxCeleb2 prepared!')

def finalize(output_folder):
    subprocess.run(['utils/utt2spk_to_spk2utt.pl', os.path.join(output_folder, 'utt2spk')], stdout=open(os.path.join(output_folder, 'spk2utt'), 'w'), stderr=subprocess.STDOUT, cwd=Settings().paths.kaldi_recipe_folder)
    result = subprocess.run(['utils/fix_data_dir.sh', output_folder], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=Settings().paths.kaldi_recipe_folder)
    print(result.stdout.decode('utf-8'))
