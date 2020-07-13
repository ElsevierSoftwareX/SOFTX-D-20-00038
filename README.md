# ASVtorch

ASVtorch is a toolkit for automatic speaker recognition.

## Main features

- Complete pipelines from audio files to speaker recognition scores
- Training of deep embedding extractors
- Fast training of i-vector extractor with GPU

## Requirements

- GPU with at least 4 GB of memory (>8GB recommended)
- Preferably a computing server with many CPU cores and ample amount of RAM
- Recent Kaldi installation
  - Needed in feature extraction and data augmentation
  - Also used for UBM training in i-vector systems
- ffmpeg
- Python environment (installation instructions below)

## Installation

1) Install ffmpeg if not yet installed
2) Install Kaldi if not yet installed
   - http://kaldi-asr.org/doc/install.html
   - Note: Augmentation scripts in Kaldi have changed over time (for example in 2019). Thus, if you encounter problems in data augmentation, try to update your Kaldi installation.
3) Install a python environment (instructions below are for conda):
   1) `conda create -n asvtorch python=3.7`
   2) `conda activate asvtorch`
   3) `conda install -c pykaldi pykaldi-cpu`
   4) Check your CUDA version with \
        `cat /usr/local/cuda/version.txt` \
        Get PyTorch installation command for the correct CUDA version from https://pytorch.org/ and install PyTorch. For example: \
        `conda install pytorch torchvision cudatoolkit=9.2 -c pytorch` 
        - Your CUDA version has to be greater than the CUDA version in PyTorch
   5) `conda install scipy matplotlib`
   6) `pip install wget`
4) Clone ASVtorch repository
   1) Navigate to a folder where you want ASVtorch folder to be placed to
   2) `git clone https://gitlab.com/ville.vestman/asvtorch.git`
   3) `cd asvtorch`
- To install updates later on:
  - run `git pull` in `asvtorch` folder

## Running the VoxCeleb recipe

- See instructions from [asvtorch/recipes/voxceleb/xvector/README.md](asvtorch/recipes/voxceleb/xvector/README.md)
- For more info on how to execute and configure experiments, see [asvtorch/src/settings/README.md](asvtorch/src/settings/README.md)
 