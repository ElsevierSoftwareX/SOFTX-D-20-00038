# VoxCeleb x-vector recipe

## Recipe description

- Feature extraction and augmentation using Kaldi
- DNN embedding extractor (x-vector or similar) training using augmented VoxCeleb2 data (segments from the same source YouTube video concatenated together)
- PLDA scoring of six VoxCeleb speaker verification trial lists
- Outputs EERs and minDCFs

## Requirements

- Kaldi installation
- MUSAN corpus (if augmentation is to be done)
- VoxCeleb 1.1 dataset
- VoxCeleb 2 dataset
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
- Change `paths.datasets` so that `voxceleb1` and `voxceleb2` are mapped to the correct locations in your system.

## Running the recipe

- Activate python environment: `conda activate asvtorch`

- To execute the recipe step-by-step run the following:
    1) `python asvtorch/recipes/voxceleb/xvector/run.py prepare`
        - This will prepare VoxCeleb1 and VoxCeleb2 datasets.
    2) `python asvtorch/recipes/voxceleb/xvector/run.py fe`
        - This will extract MFCCs for VoxCeleb1 and VoxCeleb2.
    3) `python asvtorch/recipes/voxceleb/xvector/run.py aug`
        - This will augment the training data to 5x size.
        - If the augmentation does not work or you want to skip it, you can change the network training and PLDA training data to use non-augmented data by modifying `run.py`: comment lines with `voxceleb2_cat_combined` and uncomment lines with `voxceleb2_cat` in stage 5 and later.
    4) `python asvtorch/recipes/voxceleb/xvector/run.py net`
        - This will train the default network.
    5) `python asvtorch/recipes/voxceleb/xvector/run.py emb_net`
        - This will extract embeddings for PLDA training and for trial list utterances.
    6) `python asvtorch/recipes/voxceleb/xvector/run.py score_net`
        - This will process the embeddings, train PLDA, score the trial lists, and compute EER and minDCF metrics.

- To execute the whole recipe at one go: \
    `python asvtorch/recipes/voxceleb/xvector/run.py prepare fe aug net emb_net score_net`

## Expected results

### Results in with full PLDA training data and score normalization (scoring stage)

``` txt
EER = 2.9640  minDCF = 0.2010  [epoch 37] [vox1_original]
EER = 2.8344  minDCF = 0.1985  [epoch 37] [vox1_cleaned]
EER = 2.8438  minDCF = 0.1813  [epoch 37] [vox1_extended_original]
EER = 2.7514  minDCF = 0.1792  [epoch 37] [vox1_extended_cleaned]
EER = 4.7982  minDCF = 0.2852  [epoch 37] [vox1_hard_original]
EER = 4.6833  minDCF = 0.2834  [epoch 37] [vox1_hard_cleaned]
```

### Results during network training with limited PLDA data

``` txt
EER = 7.2271  minDCF = 0.4130  [epoch 5] [vox1_original]
EER = 7.0806  minDCF = 0.4112  [epoch 5] [vox1_cleaned]
EER = 5.6735  minDCF = 0.3254  [epoch 10] [vox1_original]
EER = 5.5517  minDCF = 0.3233  [epoch 10] [vox1_cleaned]
EER = 4.8570  minDCF = 0.2966  [epoch 15] [vox1_original]
EER = 4.6849  minDCF = 0.2945  [epoch 15] [vox1_cleaned]
EER = 4.5919  minDCF = 0.2833  [epoch 20] [vox1_original]
EER = 4.4563  minDCF = 0.2811  [epoch 20] [vox1_cleaned]
EER = 4.1783  minDCF = 0.2599  [epoch 25] [vox1_original]
EER = 4.0468  minDCF = 0.2577  [epoch 25] [vox1_cleaned]
EER = 4.1571  minDCF = 0.2464  [epoch 30] [vox1_original]
EER = 4.0521  minDCF = 0.2441  [epoch 30] [vox1_cleaned]
EER = 4.1093  minDCF = 0.2485  [epoch 35] [vox1_original]
EER = 3.9989  minDCF = 0.2461  [epoch 35] [vox1_cleaned]
EER = 4.0934  minDCF = 0.2480  [epoch 37] [vox1_original]
EER = 3.9670  minDCF = 0.2457  [epoch 37] [vox1_cleaned]
```

### Expected running time

About 30 hours from the beginning to the end with the following hardware:

- 500 GB of RAM
- GeForce RTX 2048 Ti (11 GB)
- HDD for audio files, SSD for output files (features etc..)

## Notes

- Training fully-fledged ASV systems can take a long time. If the training is done in remote server and you want to prevent execution from stopping when the terminal closes or internet connection fails, consider using the following (or similar) command: \
    `nohup python -u asvtorch/recipes/voxceleb/xvector/run.py net > out.txt &`
  - `> out.txt` redirects outputs to a log file.
  - `&` runs the command in background.
  - `nohup` allows the execution to continue in server when the terminal closes.
  - `-u` prevents python from buffering outputs, so that `out.txt` gets updated without lags.
  - run `tail -f out.txt` to follow the progress.

- If the dataset is prepared and features are extracted, do not prepare datasets again without removing the folder \
    `<output_folder>/datasets/<dataset_folder>/`. \
    Otherwise you have a high change of getting errors.
