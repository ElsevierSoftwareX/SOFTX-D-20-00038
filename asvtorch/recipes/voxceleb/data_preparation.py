from typing import List, Optional
import os
import subprocess

import wget

from asvtorch.src.misc.fileutils import ensure_exists
from asvtorch.src.settings.settings import Settings
import asvtorch.src.misc.fileutils as fileutils

def prepare_datasets(datasets: Optional[List[str]]):
    dataset2func = {
        'voxceleb1': _prepare_voxceleb1,
        'voxceleb2': _prepare_voxceleb2
    }
    if datasets is None:  # If no list is given, prepare all
        datasets = dataset2func.keys()
    for dataset in datasets:
        dataset2func[dataset]()
        print(sep='\n\n')

_TRIAL_URLS = [
        'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt', 
        'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt',
        'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard.txt',
        'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt',
        'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all.txt',
        'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt']

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


def _download_trial_files():
    print('Downloading trial files for VoxCeleb...')
    output_directory = os.path.join(fileutils.get_folder_of_file(__file__), 'trial_lists')
    ensure_exists(output_directory)
    for url in _TRIAL_URLS:
        wget.download(url, out=output_directory)

def _prepare_voxceleb1():
    print('Preparing VoxCeleb1...')
    _download_trial_files()
    data_folders = (os.path.join(Settings().paths.datasets['voxceleb1'], 'wav'), )
    output_folder = fileutils.get_list_folder('voxceleb1')
    def wav_func(filename):
        return filename

    file_dict = _travel_voxceleb_folders(data_folders)
    ensure_exists(output_folder)
    wav_output = os.path.join(output_folder, 'wav.scp')
    utt2spk_output = os.path.join(output_folder, 'utt2spk')
    with open(wav_output, 'w') as wav_out, open(utt2spk_output, 'w') as utt2spk_out:
        for key in file_dict:
            for filename in file_dict[key]:
                utt_id = filename.split('/')[-1].split('.')[0]
                utt_id = '-'.join((key[0], key[1], utt_id))
                wav_out.write('{} {}\n'.format(utt_id, wav_func(filename)))
                utt2spk_out.write('{} {}\n'.format(utt_id, key[0]))

    for url in _TRIAL_URLS:
        filename = url.rsplit('/', 1)[1]
        input_file = os.path.join(os.path.join(fileutils.get_folder_of_file(__file__), 'trial_lists', filename))
        output_file = os.path.join(output_folder, filename)
        with open(input_file) as f, open(output_file, 'w') as outf:
            f.readline()
            for line in f:
                parts = line.split()
                utt_ids = []
                for fullfile in parts[1:]:
                    parts2 = fullfile.split('/')
                    utt_ids.append('-'.join((parts2[0], parts2[1], parts2[2].split('.')[0])))
                outf.write('{} {} {}\n'.format(utt_ids[0], utt_ids[1], parts[0]))

    _finalize(output_folder)
    print('VoxCeleb1 prepared!')

def _prepare_voxceleb2():
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

    file_dict = _travel_voxceleb_folders(data_folders)
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
    _finalize(output_folder)
    _finalize(output_folder_cat)

    print('VoxCeleb2 prepared!')

def _finalize(output_folder):
    subprocess.run(['utils/utt2spk_to_spk2utt.pl', os.path.join(output_folder, 'utt2spk')], stdout=open(os.path.join(output_folder, 'spk2utt'), 'w'), stderr=subprocess.STDOUT, cwd=Settings().paths.kaldi_recipe_folder)
    result = subprocess.run(['utils/fix_data_dir.sh', output_folder], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=Settings().paths.kaldi_recipe_folder)
    print(result.stdout.decode('utf-8'))
