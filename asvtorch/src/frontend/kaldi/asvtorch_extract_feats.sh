#!/bin/sh

# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.

# Modified for ASVtorch: 2020   Ville Vestman

export train_cmd="run.pl --mem 4G"
. ./path.sh
set -e

list_dir=$1
log_dir=$2
mfcc_dir=$3
config_file=$4
njobs=$5

steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config $config_file --nj $njobs --cmd "$train_cmd" $list_dir $log_dir $mfcc_dir
utils/fix_data_dir.sh $list_dir
sid/compute_vad_decision.sh --nj $njobs --cmd "$train_cmd" $list_dir $log_dir $mfcc_dir
utils/fix_data_dir.sh $list_dir

echo "Feature extraction completed!"
