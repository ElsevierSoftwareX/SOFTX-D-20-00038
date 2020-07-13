# SITW i-vector recipe


## Recipe description

- Feature extraction and UBM training using Kaldi
- UBM training using VoxCeleb1
- I-vector extractor training using VoxCeleb1 and VoxCeleb2
- PLDA scoring of `core-core` and `core-multi` trial lists
- Outputs EERs and minDCFs
- Similar to the recipe used in the following publication:
  
``` bibtex
@inproceedings{vestman2019unleashing,
author={Ville Vestman and Kong Aik Lee and Tomi H. Kinnunen and Takafumi Koshinaka},
title={{Unleashing the Unused Potential of i-Vectors Enabled by GPU Acceleration}},
year=2019,
booktitle={Proc. Interspeech 2019},
pages={351--355},
doi={10.21437/Interspeech.2019-1955},
url={http://dx.doi.org/10.21437/Interspeech.2019-1955}
}
```

Link to the publication: [http://dx.doi.org/10.21437/Interspeech.2019-1955](http://dx.doi.org/10.21437/Interspeech.2019-1955)

## Requirements

- Kaldi installation
- VoxCeleb 1.1 dataset
- VoxCeleb 2 dataset
- SITW dataset
- About 50 GB of free space for the output folder

## Recipe preparation

Update the settings in the initial config file [configs/init_config.py](configs/init_config.py) to match your system setup. Use different output folder for this recipe than for other recipes (feature configuration is different).

## Running the recipe

- Activate python environment: `conda activate asvtorch`

- To execute the recipe step-by-step run the following:
    1) `python asvtorch/recipes/sitw/ivector/run.py prepare`
        - This will prepare VoxCeleb1, VoxCeleb2, and SITW datasets.
    2) `python asvtorch/recipes/sitw/ivector/run.py fe`
        - This will extract MFCCs for VoxCeleb1, VoxCeleb2, and SITW.
    3) `python asvtorch/recipes/sitw/ivector/run.py ubm`
        - This will train the UBM using Kaldi.
    4) `python asvtorch/recipes/sitw/ivector/run.py post`
        - This will compute the component posteriors for training and trial data.
    5) `python asvtorch/recipes/sitw/ivector/run.py ivec`
        - This will train the i-vector extractor.
    6) `python asvtorch/recipes/sitw/ivector/run.py ext_ivec`
        - This will extract i-vectors for PLDA training and for trial list utterances.
    7) `python asvtorch/recipes/sitw/ivector/run.py score_ivec`
        - This will process the i-vectors, train PLDA, score the trial lists, and compute EER and minDCF metrics.

- To execute the whole recipe at one go: \
    `python asvtorch/recipes/sitw/ivector/run.py prepare fe ubm post ivec ext_ivec score_ivec`

## Expected results

``` txt
EER = 6.3150  minDCF = 0.3865  [iteration 20] [SITW core-core]
EER = 8.9497  minDCF = 0.4594  [iteration 20] [SITW core-multi]
```

### Expected running time

About 7 hours from the beginning to the end with the following hardware:

- 500 GB of RAM
- GeForce RTX 2048 Ti (11 GB)
- HDD for audio files, SSD for output files (features etc..)

## Notes

- The first iteration of the i-vector extractor training is slower than the other ones. This is because the 2nd order statistics are computed during the first iteration only. By disabling residual covariance updates, the first iteration becomes as fast as the others.

- The limiting factor of the speed of the currect i-vector implementation is the sufficient statistics computation, which happens in CPU. The longer the utterances are, the slower the statistics computation is. With short utterances such as utterances from VoxCeleb, statistics computation is fast enough to not cause considerable bottleneck. To speed up the computation, you may try to experiment with different values for `ivector_dataloader_workers` setting (larger value is not necessarily always better).

- Training fully-fledged ASV systems can take a long time. If the training is done in remote server and you want to prevent execution from stopping when the terminal closes or internet connection fails, use the following command: \
    `nohup python -u asvtorch/recipes/sitw/ivector/run.py prepare fe ubm post ivec ext_ivec score_ivec > out.txt &`
  - `> out.txt` redirects outputs to a log file.
  - `&` runs the command in background.
  - `nohup` allows the execution to continue in server when the terminal closes.
  - `-u` prevents python from buffering outputs, so that `out.txt` gets updated without lags.
  - run `tail -f out.txt` to follow the progress.

- If the dataset is prepared and features are extracted, do not prepare datasets again without removing the folder \
    `<output_folder>/datasets/<dataset_folder>/`. \
    Otherwise you have a high change of getting errors.
