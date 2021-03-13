# pylint: skip-file

computing.ubm_training_workers = 8
computing.ivector_dataloader_workers = 22
computing.feature_extraction_workers = 22
computing.use_gpu = True
computing.gpu_ids = (0,)

paths.output_folder = '/media/ssd2/vvestman/sitw_ivector_outputs'
paths.feature_and_list_folder = 'datasets'  # No need to update this
paths.kaldi_recipe_folder = '/media/hdd2/vvestman/kaldi/egs/voxceleb/v1'  # ivector recipe
paths.musan_folder = '/media/hdd3/musan'  # Used in Kaldi's augmentation
paths.datasets = { 'voxceleb1': '/media/hdd3/voxceleb1', 'voxceleb2': '/media/hdd3/voxceleb2', 'sitw': '/media/hdd3/sitw'}

features.vad_mismatch_tolerance = 0 