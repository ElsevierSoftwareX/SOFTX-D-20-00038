## Using multiple GPUs in parallel to train neural networks

Neural networks in `voxceleb/xvector` and `sitw/xvector` recipes can be trained using multiple GPUs. The toolkit achieves this by using `DistributedDataParallel` and `torch.distributed.launch` utilities of PyTorch.

Instead of the typical single GPU execution:
```txt
python asvtorch/recipes/voxceleb/xvector/run.py net
```
multi-GPU training can be started with a command:
```txt
python -m torch.distributed.launch --nproc_per_node=2 asvtorch/recipes/voxceleb/xvector/run.py net
```

Here `--nproc_per_node` specifies the number of GPUs. By default, the above command will use the first two GPUs that you have (or that are visible as determined by CUDA_VISIBLE_DEVICES environment variable). 

`Settings().computing.gpu_ids` allows you to select which GPUs will be used. For example, if you have four GPUs and want to use all except the second GPU, then set `computing.gpu_ids=(0,2,3)` and run
```txt
python -m torch.distributed.launch --nproc_per_node=3 asvtorch/recipes/voxceleb/xvector/run.py net
```
The best place to change `computing.gpu_ids` setting is in either in the `init_config.py` or `run_configs.py` file of the recipe.


## Concurrent training of multiple models

The toolkit also makes it easy to simultaneously train multiple different models (one model per GPU).

For example, in `run_configs.py` you may have:
```
net
paths.system_folder = 'system'
computing.gpu_ids=(0,)
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
network.frame_layer_size: int = 512
network.stat_size: int = 1500
network.embedding_size: int = 512

larger_net < net
paths.system_folder = 'larger_system'
computing.gpu_ids = (1,)
network.frame_layer_size = 1024
network.stat_size = 2000
```

And then you can execute back to back
```txt
python asvtorch/recipes/voxceleb/xvector/run.py net
```
and
```txt
python asvtorch/recipes/voxceleb/xvector/run.py larger_net
```

This will train two networks at the same time. The first network is trained using GPU 0 and will be saved to a relative folder called `system`. The second network is trained using GPU 1 and will be saved to a relative folder called `larger_system`.

`larger_net < net` means that `larger_net` will inherit settings from `net`. After that it will override some of the settings to make the networks different.
