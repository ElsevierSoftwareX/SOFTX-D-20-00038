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



# Extracts all features (we use "cat" version of voxceleb2 for training)
fe
recipe.start_stage = 1
recipe.end_stage = 1
recipe.feature_extraction_datasets = ['voxceleb1', 'voxceleb2_cat']

# Extracts "non-cat" VoxCeleb2 features (not needed in the recipe)
fe_vox2 < fe
recipe.feature_extraction_datasets = ['voxceleb2']



# Augments the training data (5x) --- 5x is maximum amount supported
aug
recipe.start_stage = 2
recipe.end_stage = 2
recipe.augmentation_datasets = {'voxceleb2_cat': 5}



# Trains the default network architecture
net
paths.system_folder = 'full_system_default'
recipe.start_stage = 5
recipe.end_stage = 5
network.min_clip_size = 200
network.max_clip_size = 200
network.print_interval = 500
network.weight_decay = 0.001
network.utts_per_speaker_in_epoch = 300
network.eer_stop_epochs = 5
network.max_epochs = 1000
network.initial_learning_rate = 0.2
network.min_loss_change_ratio = 0.01
network.target_loss = 0.1
network.epochs_per_train_call = 5
network.max_batch_size_in_frames = 15000
network.max_consecutive_lr_updates = 2

# To resume training from a specific epoch:
net_resume < net
network.resume_epoch = 23

# To train network with squeeze-and-excitation:
net_se < net
paths.system_folder = 'full_system_se'
network.network_class = 'asvtorch.src.networks.architectures.StandardSeNet'

# To train network with squeeze-and-excitation and residual blocks:
net_resse < net
paths.system_folder = 'full_system_resse'
network.network_class = 'asvtorch.src.networks.architectures.StandardResSeNet'

# To train network with LDE layer:
net_lde_isotropic < net
paths.system_folder = 'full_system_lde_isotropic'
network.pooling_layer_type = 'clustering'
network.cluster_assignment_mode = 'lde'
network.supervector_mode = 'lde'
network.normalize_supervector = False
network.n_clusters = 64
network.n_ghost_clusters = 0
network.stat_size = 128

# Another variant of LDE:
net_lde_tied_diagonal < net_lde_isotropic
paths.system_folder = 'full_system_lde_tied_diagonal'
network.lde_covariance_type = 'shared_diagonal'

# Network with netVLAD:
net_netvlad < net_lde_isotropic
paths.system_folder = 'full_system_netvlad'
network.cluster_assignment_mode = 'net_vlad'
network.supervector_mode = 'net_vlad'
network.normalize_supervector = True



# EMBEDDING EXTRACTION

# Not runnable as is (just to let other configs to inherit it)
emb
recipe.start_stage = 7
recipe.end_stage = 7
recipe.selected_epoch = None  # None = select the last epoch automatically
network.max_batch_size_in_frames = 30000

# Below are the embedding extraction configs for different networks

emb_net < emb < net

emb_net_se < emb < net_se

emb_net_resse < emb < net_resse

emb_net_lde_isotropic < emb < net_lde_isotropic

emb_net_lde_tied_diagonal < emb < net_lde_tied_diagonal

emb_net_netvlad < emb < net_netvlad


# SCORING

# Not runnable as is (just to let other configs to inherit it)
score
recipe.start_stage = 9
recipe.end_stage = 9
backend.plda_dim = 200
backend.score_norm_full_cohort_size = 2000
backend.score_norm_adaptive_cohort_size = 200

score_net < score < emb_net

score_net_se < score < emb_net_se

score_net_resse < score < emb_net_resse

score_net_lde_isotropic < score < emb_net_lde_isotropic

score_net_lde_tied_diagonal < score < emb_net_lde_tied_diagonal

score_net_netvlad < score < emb_net_netvlad
