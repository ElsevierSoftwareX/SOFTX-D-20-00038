# SITW x-vector recipe

## Recipe description

- Feature extraction and augmentation using Kaldi
- DNN embedding extractor (x-vector or similar) training using augmented VoxCeleb1 and VoxCeleb2 data (segments from the same source YouTube video concatenated together)
- PLDA scoring of `core-core` and `core-multi` trial lists
- Outputs EERs and minDCFs

## Requirements

- Kaldi installation
- MUSAN corpus (if augmentation is to be done)
- VoxCeleb 1.1 dataset
- VoxCeleb 2 dataset
- SITW dataset
- About 150 GB of free space for the output folder

## Recipe preparation

Update the settings in the initial config file [configs/init_config.py](configs/init_config.py):

- The computing settings below can be changed based on your system:
  - `network_dataloader_workers` is most likely not benefiting from more than 10 CPU workers. Probably 5 workers is enough most times. If the dataloader is not loading the data as fast as GPU can use it, you may try to increase this value.
  
```txt
    computing.network_dataloader_workers = 10
    computing.feature_extraction_workers = 44
    computing.use_gpu = True
    computing.gpu_ids = (0,)
```

- Change `paths.output_folder` to point to the desired output folder where all the outputs (features, lists, network models, etc...) should be stored. The folder does not need to exist beforehand.
  - Typically you would probably want to have different output folder for each recipe to avoid accidentally overridding outputs of other recipes. However, if two or more recipes share the same features and datasets, the use of the same output folder for multiple recipes makes sense.
- Change `paths.kaldi_recipe_folder` to point to `voxceleb/v2` recipe of **your** Kaldi installation.
- Change `paths.musan_folder` to point to the folder where MUSAN dataset is.
  - (used for augmentation)
- Change `paths.datasets` so that `voxceleb1`, `voxceleb2`, and `sitw` are mapped to the correct locations in your system.

## Running the recipe

- Activate python environment: `conda activate asvtorch`

- To execute the recipe step-by-step run the following:
    1) `python asvtorch/recipes/sitw/xvector/run.py prepare`
        - This will prepare the datasets.
    2) `python asvtorch/recipes/sitw/xvector/run.py fe`
        - This will extract MFCCs for the datasets.
    3) `python asvtorch/recipes/sitw/xvector/run.py aug`
        - This will augment the training data to 5x size.
    4) `python asvtorch/recipes/sitw/xvector/run.py net`
        - This will train the default network.
    5) `python asvtorch/recipes/sitw/xvector/run.py emb_net`
        - This will extract embeddings for PLDA training and for trial list utterances.
    6) `python asvtorch/recipes/sitw/xvector/run.py score_net`
        - This will process the embeddings, train PLDA, score the trial lists, apply score normalization, and compute EER and minDCF metrics.

- To execute the whole recipe at one go: \
    `python asvtorch/recipes/sitw/xvector/run.py prepare fe aug net emb_net score_net`

## Expected results

### Results in with full PLDA training data and score normalization (scoring stage)

``` txt
EER = 2.2173  minDCF = 0.1511  [epoch 39] [SITW core-core]
EER = 3.7730  minDCF = 0.2130  [epoch 39] [SITW core-multi]
```

### Results during network training with limited PLDA data

``` txt
EER = 4.9195  minDCF = 0.3113  [epoch 5] [SITW core-core]
EER = 6.8872  minDCF = 0.3717  [epoch 5] [SITW core-multi]
EER = 3.6905  minDCF = 0.2452  [epoch 10] [SITW core-core]
EER = 5.5668  minDCF = 0.3131  [epoch 10] [SITW core-multi]
EER = 3.0344  minDCF = 0.2216  [epoch 15] [SITW core-core]
EER = 5.1964  minDCF = 0.2922  [epoch 15] [SITW core-multi]
EER = 2.9261  minDCF = 0.2145  [epoch 20] [SITW core-core]
EER = 4.9677  minDCF = 0.2853  [epoch 20] [SITW core-multi]
EER = 2.8415  minDCF = 0.2125  [epoch 25] [SITW core-core]
EER = 4.8283  minDCF = 0.2835  [epoch 25] [SITW core-multi]
EER = 2.8661  minDCF = 0.2067  [epoch 30] [SITW core-core]
EER = 4.8804  minDCF = 0.2810  [epoch 30] [SITW core-multi]
EER = 2.8474  minDCF = 0.2042  [epoch 35] [SITW core-core]
EER = 4.8482  minDCF = 0.2782  [epoch 35] [SITW core-multi]
EER = 2.8704  minDCF = 0.2053  [epoch 39] [SITW core-core]
EER = 4.8777  minDCF = 0.2782  [epoch 39] [SITW core-multi]
```

### Expected running time

About 35 hours from the beginning to the end with the following hardware:

- 500 GB of RAM
- GeForce RTX 2048 Ti (11 GB)
- HDD for audio files, SSD for output files (features etc..)

## Notes

- Training fully-fledged ASV systems can take a long time. If the training is done in remote server and you want to prevent execution from stopping when the terminal closes or internet connection fails, consider using the following (or similar) command: \
    `nohup python -u asvtorch/recipes/sitw/xvector/run.py net > out.txt &`
  - `> out.txt` redirects outputs to a log file.
  - `&` runs the command in background.
  - `nohup` allows the execution to continue in server when the terminal closes.
  - `-u` prevents python from buffering outputs, so that `out.txt` gets updated without lags.
  - run `tail -f out.txt` to follow the progress.

- If the dataset is prepared and features are extracted, do not prepare datasets again without removing the folder \
    `<output_folder>/datasets/<dataset_folder>/`. \
    Otherwise you have a high change of getting errors.
