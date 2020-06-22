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
    computing.gpu_id = 0
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
EER = 2.8739  minDCF = 0.1932  [epoch 7] [vox1_original]
EER = 2.7120  minDCF = 0.1907  [epoch 7] [vox1_cleaned]
EER = 2.8840  minDCF = 0.1869  [epoch 7] [vox1_extended_original]
EER = 2.7728  minDCF = 0.1846  [epoch 7] [vox1_extended_cleaned]
EER = 5.0285  minDCF = 0.2930  [epoch 7] [vox1_hard_original]
EER = 4.9167  minDCF = 0.2912  [epoch 7] [vox1_hard_cleaned]
```

### Results during network training with limited PLDA data

``` txt
EER = 5.7690  minDCF = 0.3392  [epoch 1] [vox1_original]
EER = 5.6262  minDCF = 0.3373  [epoch 1] [vox1_cleaned]
EER = 4.8994  minDCF = 0.3115  [epoch 2] [vox1_original]
EER = 4.7434  minDCF = 0.3083  [epoch 2] [vox1_cleaned]
EER = 4.6714  minDCF = 0.2991  [epoch 3] [vox1_original]
EER = 4.5573  minDCF = 0.2970  [epoch 3] [vox1_cleaned]
EER = 4.3479  minDCF = 0.2858  [epoch 4] [vox1_original]
EER = 4.2436  minDCF = 0.2827  [epoch 4] [vox1_cleaned]
EER = 4.3214  minDCF = 0.2795  [epoch 5] [vox1_original]
EER = 4.1851  minDCF = 0.2763  [epoch 5] [vox1_cleaned]
EER = 4.1783  minDCF = 0.2687  [epoch 6] [vox1_original]
EER = 4.0149  minDCF = 0.2665  [epoch 6] [vox1_cleaned]
EER = 4.1199  minDCF = 0.2568  [epoch 7] [vox1_original]
EER = 3.9936  minDCF = 0.2545  [epoch 7] [vox1_cleaned]
```

### Expected running time

About 36 hours from the beginning to the end with the following hardware:

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
