# VoxCeleb neural i-vector recipe

## Recipe description

- Feature extraction and augmentation using Kaldi
- DNN embedding extractor (x-vector or similar) training using augmented VoxCeleb2 data (segments from the same source YouTube video concatenated together)
- Neural i-vector training using sufficient statistics extracted from the network
- PLDA scoring of six VoxCeleb speaker verification trial lists
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
- About 200 GB of free space for the output folder

## Recipe preparation

Update the settings in the initial config file [configs/init_config.py](configs/init_config.py) to match your system setup. \
**Note**: You can use the same output folder with standard x-vector recipe (same features are used). If feature extraction and augmentation is already done, **you can continue running this recipe from stage 5 (network training) onwards**.

## Running the recipe

- Activate python environment: `conda activate asvtorch`

- To execute the recipe step-by-step run the following:
    1) `python asvtorch/recipes/voxceleb/neural_ivector/run.py prepare`
        - This will prepare VoxCeleb1 and VoxCeleb2 datasets.
    2) `python asvtorch/recipes/voxceleb/neural_ivector/run.py fe`
        - This will extract MFCCs for VoxCeleb1 and VoxCeleb2.
    3) `python asvtorch/recipes/voxceleb/neural_ivector/run.py aug`
        - This will augment the training data to 5x size.
        - If the augmentation does not work or you want to skip it, you can change the network training and PLDA training data to use non-augmented data by modifying `run.py`: comment lines with `voxceleb2_cat_combined` and uncomment lines with `voxceleb2_cat` in stage 5 and later.
    4) `python asvtorch/recipes/voxceleb/neural_ivector/run.py net_lde_tied_diagonal`
        - This will train the network with LDE layer.
    5) `python asvtorch/recipes/voxceleb/neural_ivector/run.py emb_net_lde_tied_diagonal` (optional)
        - This will extract embeddings for PLDA training and for trial list utterances.
    6) `python asvtorch/recipes/voxceleb/neural_ivector/run.py score_net_lde_tied_diagonal` (optional)
        - This will process the embeddings, train PLDA, score the trial lists, apply score normalization, and compute EER and minDCF metrics.
    7) `python asvtorch/recipes/voxceleb/neural_ivector/run.py stats`
        - This will extract Baum-Welch statistics from the network for i-vector extraction.
    8) `python asvtorch/recipes/voxceleb/neural_ivector/run.py ivec`
        - This will train the neural i-vector extractor.
    9) `python asvtorch/recipes/voxceleb/neural_ivector/run.py ext_ivec`
        - This will extract neural i-vectors for PLDA training and for trial utterances.
    10) `python asvtorch/recipes/voxceleb/neural_ivector/run.py score_ivec`
        - This will process the i-vectors, train PLDA, score the trial lists, apply score normalization, and compute EER and minDCF metrics.

- To execute the neural-ivector recipe from stage 5 onwards: \
    `python asvtorch/recipes/voxceleb/neural_ivector/run.py net_lde_tied_diagonal emb_net_lde_tied_diagonal score_net_lde_tied_diagonal stats ivec ext_ivec score_ivec`

## Expected results

### Neural i-vector results

``` txt
EER = 4.6396  minDCF = 0.2889  [iteration 5] [vox1_original]
EER = 4.4829  minDCF = 0.2866  [iteration 5] [vox1_cleaned]
EER = 4.4693  minDCF = 0.2813  [iteration 5] [vox1_extended_original]
EER = 4.3716  minDCF = 0.2794  [iteration 5] [vox1_extended_cleaned]
EER = 7.4797  minDCF = 0.4171  [iteration 5] [vox1_hard_original]
EER = 7.3724  minDCF = 0.4154  [iteration 5] [vox1_hard_cleaned]
```

### Results for LDE embeddings

``` txt
EER = 3.0170  minDCF = 0.2001  [epoch 40] [vox1_original]
EER = 2.8609  minDCF = 0.1977  [epoch 40] [vox1_cleaned]
EER = 2.9246  minDCF = 0.1858  [epoch 40] [vox1_extended_original]
EER = 2.8169  minDCF = 0.1838  [epoch 40] [vox1_extended_cleaned]
EER = 4.9086  minDCF = 0.2939  [epoch 40] [vox1_hard_original]
EER = 4.8053  minDCF = 0.2917  [epoch 40] [vox1_hard_cleaned]
```

### Results during network training with limited PLDA data

``` txt
EER = 6.9938  minDCF = 0.4108  [epoch 5] [vox1_original]
EER = 6.8519  minDCF = 0.4090  [epoch 5] [vox1_cleaned]
EER = 5.9705  minDCF = 0.3545  [epoch 10] [vox1_original]
EER = 5.8495  minDCF = 0.3525  [epoch 10] [vox1_cleaned]
EER = 5.2387  minDCF = 0.3216  [epoch 15] [vox1_original]
EER = 5.0997  minDCF = 0.3195  [epoch 15] [vox1_cleaned]
EER = 4.7721  minDCF = 0.2860  [epoch 20] [vox1_original]
EER = 4.6211  minDCF = 0.2837  [epoch 20] [vox1_cleaned]
EER = 4.4646  minDCF = 0.2732  [epoch 25] [vox1_original]
EER = 4.3552  minDCF = 0.2709  [epoch 25] [vox1_cleaned]
EER = 4.3585  minDCF = 0.2697  [epoch 30] [vox1_original]
EER = 4.2170  minDCF = 0.2675  [epoch 30] [vox1_cleaned]
EER = 4.3585  minDCF = 0.2618  [epoch 35] [vox1_original]
EER = 4.2489  minDCF = 0.2585  [epoch 35] [vox1_cleaned]
EER = 4.3532  minDCF = 0.2653  [epoch 40] [vox1_original]
EER = 4.1957  minDCF = 0.2630  [epoch 40] [vox1_cleaned]
```

## Notes

- Training fully-fledged ASV systems can take a long time. If the training is done in remote server and you want to prevent execution from stopping when the terminal closes or internet connection fails, consider using the following (or similar) command: \
    `nohup python -u asvtorch/recipes/voxceleb/neural_ivector/run.py net_lde_tied_diagonal > out.txt &`
  - `> out.txt` redirects outputs to a log file.
  - `&` runs the command in background.
  - `nohup` allows the execution to continue in server when the terminal closes.
  - `-u` prevents python from buffering outputs, so that `out.txt` gets updated without lags.
  - run `tail -f out.txt` to follow the progress.

- If the dataset is prepared and features are extracted, do not prepare datasets again without removing the folder \
    `<output_folder>/datasets/<dataset_folder>/`. \
    Otherwise you have a high change of getting errors.
