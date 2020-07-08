# pylint: skip-file

# Prepares all datasets
prepare
recipe.start_stage = 0
recipe.end_stage = 0

# Prepare vox1
prepare_vox1 < prepare
recipe.preparation_datasets = ['voxceleb1']

# Prepare vox2
prepare_vox2 < prepare
recipe.preparation_datasets = ['voxceleb2']

prepare_librispeech < prepare
recipe.preparation_datasets = ['librispeech']


# Extracts all features (This time we use "non-concatenated" version of voxceleb2 for training)
fe
recipe.start_stage = 1
recipe.end_stage = 1
recipe.feature_extraction_datasets = ['voxceleb1', 'voxceleb2']

# Extracts "cat" VoxCeleb2 features (not needed in the recipe)
fe_vox2_cat < fe
recipe.feature_extraction_datasets = ['voxceleb2_cat']

fe_librispeech < fe
recipe.feature_extraction_datasets = ['librispeech']


# Augments the training data (5x) --- 5x is maximum amount supported
# aug
# recipe.start_stage = 2
# recipe.end_stage = 2
# recipe.augmentation_datasets = {'voxceleb2_cat': 5}


ubm
recipe.start_stage = 3
recipe.end_stage = 3
features.cmvn = 'm'  # m = mean normalization, mv = mean & variance normalization, or empty for no cmvn
features.cmvn_window = 300  # Sliding window size in frames
features.deltas = 'bva'  # b = base coefficients, v = deltas (velocity), a = double deltas (acceleration)
features.delta_reach = 3  # delta window reach in one direction
ivector.num_gaussians = 2048
ivector.ubm_name = 'ubm_2048'

# component posterior extraction for frames
post < ubm
recipe.start_stage = 4
recipe.end_stage = 4

# i-vector extractor training
ivec < ubm
recipe.start_stage = 5
recipe.end_stage = 5
paths.system_folder = 'ivector_400'
ivector.ivec_dim = 400
ivector.ivec_type = 'augmented' # or 'standard'
ivector.update_covariances = True
ivector.n_iterations = 20

# i-vector extraction
ext_ivec < ivec
recipe.start_stage = 7
recipe.end_stage = 7
recipe.selected_iteration = 20

alexey_ivec < ivec
recipe.start_stage = 10
recipe.end_stage = 10
recipe.selected_iteration = 20

ivec_1 < alexey_ivec
recipe.crop_duration = 1

ivec_100 < alexey_ivec
recipe.crop_duration = 100

ivec_200 < alexey_ivec
recipe.crop_duration = 200

ivec_400 < alexey_ivec
recipe.crop_duration = 400

ivec_800 < alexey_ivec
recipe.crop_duration = 800

ivec_full < alexey_ivec


# SCORING
score_ivec < ext_ivec
recipe.start_stage = 9
recipe.end_stage = 9
backend.plda_dim = 300
backend.plda_rcond = 1e-7
#backend.score_norm_full_cohort_size = 2000
#backend.score_norm_adaptive_cohort_size = 200
