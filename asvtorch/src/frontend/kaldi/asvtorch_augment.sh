#!/bin/bash

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
sampling_rate=$6
number_of_aug=$7


frame_shift=0.01
awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $list_dir/lists/utt2num_frames > $list_dir/lists/reco2dur

if [ ! -d "RIRS_NOISES" ]; then
# Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
unzip rirs_noises.zip
fi

# Make a version with reverberated speech
rvb_opts=()
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

# Make a reverberated version of the list.  Note that we don't add any
# additive noise here.
python steps/data/reverberate_data_dir.py \
"${rvb_opts[@]}" \
--speech-rvb-probability 1 \
--pointsource-noise-addition-probability 0 \
--isotropic-noise-addition-probability 0 \
--num-replications 1 \
--source-sampling-rate $sampling_rate \
$list_dir/lists ${list_dir}_reverb/lists
cp $list_dir/lists/vad.scp ${list_dir}_reverb/lists
utils/copy_data_dir.sh --utt-suffix "-reverb" ${list_dir}_reverb/lists ${list_dir}_reverb.new/lists
rm -rf ${list_dir}_reverb
mv ${list_dir}_reverb.new ${list_dir}_reverb


# Augment with musan_noise
python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$list_dir/../musan_noise/lists" $list_dir/lists ${list_dir}_noise/lists
# Augment with musan_music
python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$list_dir/../musan_music/lists" $list_dir/lists ${list_dir}_music/lists
# Augment with musan_speech
python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$list_dir/../musan_speech/lists" $list_dir/lists ${list_dir}_babble/lists

# Combine reverb, noise, music, and babble into one directory.
utils/combine_data.sh ${list_dir}_aug/lists ${list_dir}_reverb/lists ${list_dir}_noise/lists ${list_dir}_music/lists ${list_dir}_babble/lists

# Take a random subset of the augmentations (128k is somewhat larger than twice
# the size of the SWBD+SRE list)
utils/subset_data_dir.sh ${list_dir}_aug/lists $number_of_aug ${list_dir}_aug_subset/lists
utils/fix_data_dir.sh ${list_dir}_aug_subset/lists

# Make filterbanks for the augmented data.  Note that we do not compute a new
# vad.scp file here.  Instead, we use the vad.scp from the clean version of
# the list.
steps/make_mfcc.sh --mfcc-config $config_file --nj $njobs --cmd "$train_cmd" ${list_dir}_aug_subset/lists $log_dir $mfcc_dir

# Combine the clean and augmented SWBD+SRE list.  This is now roughly
# double the size of the original clean list.
utils/combine_data.sh ${list_dir}_combined/lists ${list_dir}_aug_subset/lists $list_dir/lists

echo "Data augmentation completed!"
