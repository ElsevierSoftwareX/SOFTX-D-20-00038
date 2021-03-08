## Preparation of datasets

In the dataset preparation step, you will need to create the following files:
1) `wav.scp`
2) `utt2spk`
3) trial lists

The first one is responsible for mapping utterance ids (specified by you) to the corresponding audio files in the disk. The format is the same as in Kaldi Toolkit. For example, a line in `wav.scp` could look like this:
``` txt 
id10482-M8hvFMX1xEA-00024 /media/hdd3/voxceleb1/wav/id10482/M8hvFMX1xEA/00024.wav
```
If the audio file format is something else than a Kaldi-compatible wav file, then instead of specifying the path to the audio file, you should specify a command that converts the file into wav format and then directs the output the pipe (an example can be found from [asvtorch/recipes/voxceleb/data_preparation.py](asvtorch/recipes/voxceleb/data_preparation.py) [see voxceleb2 preparation])


The `utt2spk` file is responsible for mapping utterance ids to speaker ids. The format is the same is in the Kaldi Toolkit. When sorting utt2spk file, then also the speakers of these utterances should be sorted. This can be achieved by appending the speaker ids in the beginning of the utterance ids. An example of a line in `utt2spk` could be:
``` txt
id10482-M8hvFMX1xEA-00024 id10482
```
where `id10482` is the speaker id and `id10482-M8hvFMX1xEA-00024` is the utterance id.

Both `utt2spk` and `wav.scp` should be saved to a location given by  
``` python 
fileutils.get_list_folder('voxceleb1')
```
where `voxceleb1` should be replaced by the name of the dataset (given by you). The same name should be used elsewhere in the code and settings to refer to this specific dataset.

When these files are ready, you finally have to call
``` python
subprocess.run(['utils/utt2spk_to_spk2utt.pl', os.path.join(output_folder, 'utt2spk')], stdout=open(os.path.join(output_folder, 'spk2utt'), 'w'), stderr=subprocess.STDOUT, cwd=Settings().paths.kaldi_recipe_folder)
    result = subprocess.run(['utils/fix_data_dir.sh', output_folder], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=Settings().paths.kaldi_recipe_folder)
```
to finalize the dataset preparation process.


### Trial lists
The trial lists have a format
``` txt
utteranceId utteranceId2 label
```
where `label` can be `Target`, `target`, `TARGET`, `1`, `t`, or `T` to indicate that the speaker is the same in the two utterances, and `Nontarget`, `nontarget`, `NONTARGET`, `Non-target`, `non-target`, `NON-TARGET`, `0`, `n`, `N`, `f`, or `F` to indicate that the utterances are from two different speakers.

If you want to score multiple utterances against one or multiple utterances, you can do this by using an ampersand (`&`). For example,
``` txt
firstEnrollmentId&secondEnrollmentId&thirdEnrollmentId testId label
```

The saving location of trial lists is also obtained by calling 
``` python
fileutils.get_list_folder('voxceleb1')
```
where `voxceleb1` should be replaced by the name of the dataset at hand. The filename of the trial list itself can be anything.

To see a working example of dataset preparation see [asvtorch/recipes/voxceleb/data_preparation.py](asvtorch/recipes/voxceleb/data_preparation.py)