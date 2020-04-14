#!/bin/bash

# Copyright 2015   David Snyder
#           2019   Phani Sankar Nidadavolu
# Apache 2.0.
#
# This script creates the MUSAN data directory.
# Consists of babble, music and noise files.
# Used to create augmented data
# The required dataset is freely available at http://www.openslr.org/17/

# The corpus can be cited as follows:
# @misc{musan2015,
#  author = {David Snyder and Guoguo Chen and Daniel Povey},
#  title = {{MUSAN}: {A} {M}usic, {S}peech, and {N}oise {C}orpus},
#  year = {2015},
#  eprint = {1510.08484},
#  note = {arXiv:1510.08484v1}
# }

# Modified for ASVtorch: 2020   Ville Vestman


set -e
in_dir=$1
data_dir=$2
sampling_rate=$3
use_vocals=true

mkdir -p local/musan.tmp

# The below script will create the musan corpus
steps/data/make_musan.py --use-vocals ${use_vocals} \
                        --sampling-rate ${sampling_rate} \
                        ${in_dir} ${data_dir}/lists || exit 1;

utils/fix_data_dir.sh ${data_dir}/lists

grep "music" ${data_dir}/lists/utt2spk > local/musan.tmp/utt2spk_music
grep "speech" ${data_dir}/lists/utt2spk > local/musan.tmp/utt2spk_speech
grep "noise" ${data_dir}/lists/utt2spk > local/musan.tmp/utt2spk_noise

utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_music \
  ${data_dir}/lists ${data_dir}_music/lists
utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_speech \
  ${data_dir}/lists ${data_dir}_speech/lists
utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_noise \
  ${data_dir}/lists ${data_dir}_noise/lists

utils/fix_data_dir.sh ${data_dir}_music/lists
utils/fix_data_dir.sh ${data_dir}_speech/lists
utils/fix_data_dir.sh ${data_dir}_noise/lists

rm -rf local/musan.tmp

for name in speech noise music; do
    utils/data/get_reco2dur.sh ${data_dir}_${name}/lists
done
