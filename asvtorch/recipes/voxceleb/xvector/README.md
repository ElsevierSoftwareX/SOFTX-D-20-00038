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
    computing.gpu_id = 0
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
EER = 2.7042  minDCF = 0.1845  [epoch 59] [vox1_original]
EER = 2.5844  minDCF = 0.1820  [epoch 59] [vox1_cleaned]
EER = 2.7960  minDCF = 0.1764  [epoch 59] [vox1_extended_original]
EER = 2.6879  minDCF = 0.1742  [epoch 59] [vox1_extended_cleaned]
EER = 4.7078  minDCF = 0.2809  [epoch 59] [vox1_hard_original]
EER = 4.5947  minDCF = 0.2789  [epoch 59] [vox1_hard_cleaned]
```

### Results during network training with limited PLDA data

``` txt
EER = 8.7383  minDCF = 0.4522  [epoch 5] [vox1_original]
EER = 8.6227  minDCF = 0.4505  [epoch 5] [vox1_cleaned]
EER = 6.6704  minDCF = 0.3873  [epoch 10] [vox1_original]
EER = 6.5062  minDCF = 0.3844  [epoch 10] [vox1_cleaned]
EER = 5.3236  minDCF = 0.3064  [epoch 15] [vox1_original]
EER = 5.1848  minDCF = 0.3032  [epoch 15] [vox1_cleaned]
EER = 4.3373  minDCF = 0.2811  [epoch 20] [vox1_original]
EER = 4.1744  minDCF = 0.2788  [epoch 20] [vox1_cleaned]
EER = 3.7223  minDCF = 0.2386  [epoch 25] [vox1_original]
EER = 3.5948  minDCF = 0.2360  [epoch 25] [vox1_cleaned]
EER = 3.2769  minDCF = 0.2070  [epoch 30] [vox1_original]
EER = 3.1375  minDCF = 0.2045  [epoch 30] [vox1_cleaned]
EER = 3.2504  minDCF = 0.1961  [epoch 35] [vox1_original]
EER = 3.1002  minDCF = 0.1935  [epoch 35] [vox1_cleaned]
EER = 3.2132  minDCF = 0.2003  [epoch 40] [vox1_original]
EER = 3.0949  minDCF = 0.1969  [epoch 40] [vox1_cleaned]
EER = 3.1920  minDCF = 0.1935  [epoch 45] [vox1_original]
EER = 3.0524  minDCF = 0.1902  [epoch 45] [vox1_cleaned]
EER = 3.0754  minDCF = 0.1945  [epoch 50] [vox1_original]
EER = 2.9567  minDCF = 0.1913  [epoch 50] [vox1_cleaned]
EER = 3.1390  minDCF = 0.1931  [epoch 55] [vox1_original]
EER = 3.0098  minDCF = 0.1899  [epoch 55] [vox1_cleaned]
EER = 3.1178  minDCF = 0.1930  [epoch 59] [vox1_original]
EER = 2.9726  minDCF = 0.1898  [epoch 59] [vox1_cleaned]
```

### Expected running time

About 40 hours from the beginning to the end with the following hardware:

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
