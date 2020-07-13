# SITW neural i-vector recipe

## Recipe description

- Feature extraction and augmentation using Kaldi
- DNN embedding extractor (x-vector or similar) training using augmented VoxCeleb1 and VoxCeleb2 data (segments from the same source YouTube video concatenated together)
- Neural i-vector training using sufficient statistics extracted from the network
- PLDA scoring of `core-core` and `core-multi` trial lists
- Outputs EERs and minDCFs
- Similar to the recipe used in the following publication:

``` bibtex
@inproceedings{vestman2020neural,
author={Ville Vestman and Kong Aik Lee and Tomi H. Kinnunen},
title={{Neural i-vectors}},
year=2020,
booktitle={Odyssey 2020 (accepted)},
}
```

Link to the publication: [https://arxiv.org/pdf/2004.01559](https://arxiv.org/pdf/2004.01559)

## Requirements

- Kaldi installation
- MUSAN corpus (if augmentation is to be done)
- VoxCeleb 1.1 dataset
- VoxCeleb 2 dataset
- SITW dataset
- About 200 GB of free space for the output folder

## Recipe preparation

Update the settings in the initial config file [configs/init_config.py](configs/init_config.py) to match your system setup. \
**Note**: You can use the same output folder with standard x-vector recipe (same features are used). If feature extraction and augmentation is already done, **you can continue running this recipe from stage 5 (network training) onwards**.

## Running the recipe

- Activate python environment: `conda activate asvtorch`

- To execute the recipe step-by-step run the following:
    1) `python asvtorch/recipes/sitw/neural_ivector/run.py prepare`
        - This will prepare VoxCeleb1, VoxCeleb2, and SITW datasets.
    2) `python asvtorch/recipes/sitw/neural_ivector/run.py fe`
        - This will extract MFCCs for VoxCeleb1, VoxCeleb2, and SITW.
    3) `python asvtorch/recipes/sitw/neural_ivector/run.py aug`
        - This will augment the training data to 5x size.
    4) `python asvtorch/recipes/sitw/neural_ivector/run.py net_lde_tied_diagonal`
        - This will train the network with LDE layer.
    5) `python asvtorch/recipes/sitw/neural_ivector/run.py emb_net_lde_tied_diagonal` (optional)
        - This will extract embeddings for PLDA training and for trial list utterances.
    6) `python asvtorch/recipes/sitw/neural_ivector/run.py score_net_lde_tied_diagonal` (optional)
        - This will process the embeddings, train PLDA, score the trial lists, apply score normalization, and compute EER and minDCF metrics.
    7) `python asvtorch/recipes/sitw/neural_ivector/run.py stats`
        - This will extract Baum-Welch statistics from the network for i-vector extraction.
    8) `python asvtorch/recipes/sitw/neural_ivector/run.py ivec`
        - This will train the neural i-vector extractor.
    9) `python asvtorch/recipes/sitw/neural_ivector/run.py ext_ivec`
        - This will extract neural i-vectors for PLDA training and for trial utterances.
    10) `python asvtorch/recipes/sitw/neural_ivector/run.py score_ivec`
        - This will process the i-vectors, train PLDA, score the trial lists, apply score normalization, and compute EER and minDCF metrics.

- To execute the neural-ivector recipe from stage 5 onwards: \
    `python asvtorch/recipes/sitw/neural_ivector/run.py net_lde_tied_diagonal emb_net_lde_tied_diagonal score_net_lde_tied_diagonal stats ivec ext_ivec score_ivec`

## Expected results

### Neural i-vector results

``` txt
EER = 3.5028  minDCF = 0.2292  [iteration 5] [SITW core-core]
EER = 5.6147  minDCF = 0.3067  [iteration 5] [SITW core-multi]
```

### Results for LDE embeddings

``` txt
EER = 2.2964  minDCF = 0.1465  [epoch 39] [SITW core-core]
EER = 3.8228  minDCF = 0.2085  [epoch 39] [SITW core-multi]
```

### Results during network training with limited PLDA data

``` txt
EER = 4.7567  minDCF = 0.2814  [epoch 5] [SITW core-core]
EER = 6.6777  minDCF = 0.3425  [epoch 5] [SITW core-multi]
EER = 3.8546  minDCF = 0.2295  [epoch 10] [SITW core-core]
EER = 5.8238  minDCF = 0.2984  [epoch 10] [SITW core-multi]
EER = 3.4718  minDCF = 0.2114  [epoch 15] [SITW core-core]
EER = 5.4953  minDCF = 0.2800  [epoch 15] [SITW core-multi]
EER = 3.1985  minDCF = 0.2018  [epoch 20] [SITW core-core]
EER = 5.3837  minDCF = 0.2737  [epoch 20] [SITW core-multi]
EER = 3.0640  minDCF = 0.2008  [epoch 25] [SITW core-core]
EER = 5.1971  minDCF = 0.2692  [epoch 25] [SITW core-multi]
EER = 3.1164  minDCF = 0.2020  [epoch 30] [SITW core-core]
EER = 5.0572  minDCF = 0.2683  [epoch 30] [SITW core-multi]
EER = 3.0624  minDCF = 0.1999  [epoch 35] [SITW core-core]
EER = 5.1170  minDCF = 0.2654  [epoch 35] [SITW core-multi]
EER = 3.0891  minDCF = 0.2017  [epoch 39] [SITW core-core]
EER = 5.0771  minDCF = 0.2666  [epoch 39] [SITW core-multi]
```

## Notes

- Training fully-fledged ASV systems can take a long time. If the training is done in remote server and you want to prevent execution from stopping when the terminal closes or internet connection fails, consider using the following (or similar) command: \
    `nohup python -u asvtorch/recipes/sitw/neural_ivector/run.py net_lde_tied_diagonal > out.txt &`
  - `> out.txt` redirects outputs to a log file.
  - `&` runs the command in background.
  - `nohup` allows the execution to continue in server when the terminal closes.
  - `-u` prevents python from buffering outputs, so that `out.txt` gets updated without lags.
  - run `tail -f out.txt` to follow the progress.

- If the dataset is prepared and features are extracted, do not prepare datasets again without removing the folder \
    `<output_folder>/datasets/<dataset_folder>/`. \
    Otherwise you have a high change of getting errors.
