# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
import time
import random
import sys
import builtins

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import torch.optim as optim

from asvtorch.src.networks.training_dataloader import get_dataloader as get_training_dataloader
from asvtorch.src.networks.network_testing import get_dataloader as get_validation_dataloader, compute_losses_and_accuracies
import asvtorch.src.networks.network_io as network_io
from asvtorch.src.settings.settings import Settings
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader
import asvtorch.src.misc.fileutils as fileutils


def train_network(training_data: UtteranceList, resume_epoch: int = 0):

    settings = Settings().network

    training_data.convert_labels_to_numeric()

    n_speakers = training_data.get_number_of_speakers()
    print('Number of speakers: {}'.format(n_speakers))

    feat_dim = FeatureLoader().get_feature_dimension(training_data[0])

    # Training & validation:
    training_data, validation_data = _split_to_train_and_validation(training_data, settings.validation_utterances)

    # Subset of training:
    training_data_subset = _select_random_subset(training_data, settings.validation_utterances)

    training_dataloader = get_training_dataloader(training_data)
 
    validation_dataloader_1 = get_validation_dataloader(training_data_subset)
    validation_dataloader_2 = get_validation_dataloader(validation_data)

    net = network_io.initialize_net(feat_dim, n_speakers)
    net.to(Settings().computing.device)
    net_module = net
    gpu_id = Settings().computing.local_gpu_id
    if(Settings().computing.world_size > 1):
        net = DistributedDataParallel(net, device_ids=[gpu_id], output_device=gpu_id)
        net_module = net.module
        
    print_learnable_parameters(net)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print_once('Number of trainable parameters: {}'.format(total_params))

    criterion = nn.CrossEntropyLoss()

    optimizer = _init_optimizer(net, settings)

    log_folder = fileutils.get_network_log_folder()
    network_folder = fileutils.get_network_folder()
    output_filename = os.path.join(network_folder, 'epoch')

    if resume_epoch < 0:
        resume_epoch = -resume_epoch
        print_once('Computing ASV metrics for epoch {}...'.format(resume_epoch))
        network_io.load_state(output_filename, resume_epoch, net, optimizer, Settings().computing.device)
        return net, True, resume_epoch
    elif resume_epoch > 0:
        print_once('Resuming network training from epoch {}...'.format(resume_epoch))
        network_io.load_state(output_filename, resume_epoch, net, optimizer, Settings().computing.device)

    net.train()

    for epoch in range(1, settings.epochs_per_train_call + 1):

        current_learning_rate = optimizer.param_groups[0]['lr']

        logfilename = os.path.join(log_folder, 'gpu{}_epoch.{}.log'.format(gpu_id, epoch + resume_epoch))
        logfile = open(logfilename, 'w')
        print('GPU {}: Log file created: {}'.format(gpu_id, logfilename))

        print('GPU {}: Shuffling training data...'.format(gpu_id))
        training_dataloader.dataset.shuffle_and_rebatch(epoch+resume_epoch)

        training_loss = 0
        optimizer.zero_grad()

        if(Settings().computing.world_size > 1):
            torch.distributed.barrier()  # Sync processes before timing for nicer outputs
        start_time = time.time()

        # For automatic learning rate scheduling:
        losses = []
        print('GPU {}: Iterating over training minibatches...'.format(gpu_id))
         
        for i, batch in enumerate(training_dataloader):

            # Copying data to GPU:
            features, speaker_labels = batch
            features = features.to(Settings().computing.device)
            speaker_labels = speaker_labels.to(Settings().computing.device)

            outputs = net(features)
            loss = (criterion(outputs, speaker_labels)) / settings.optimizer_step_interval
            loss.backward()

            # Updating weights:
            if i % settings.optimizer_step_interval == settings.optimizer_step_interval - 1:
                optimizer.step()
                optimizer.zero_grad()

            minibatch_loss = loss.item() * settings.optimizer_step_interval
            losses.append(minibatch_loss)
            training_loss += minibatch_loss

            # Computing train and test accuracies and printing status:
            if i % settings.print_interval == settings.print_interval - 1:
                if i % (settings.print_interval * settings.accuracy_print_interval) == settings.print_interval * settings.accuracy_print_interval - 1:
                    torch.cuda.empty_cache()
                    val_data = compute_losses_and_accuracies((validation_dataloader_1, validation_dataloader_2), net, criterion)
                    output = 'GPU {}: Epoch {}, Time: {:.0f} s, Batch {}/{}, lr: {:.6f}, train-loss: {:.3f}, subset-loss: {:.3f}, val-loss: {:.3f}, subset-acc: {:.3f} val-acc: {:.3f} '.format(gpu_id, epoch + resume_epoch, time.time() - start_time, i + 1, len(training_dataloader), current_learning_rate, training_loss / settings.print_interval, val_data[0][0], val_data[1][0], val_data[0][1], val_data[1][1])
                    torch.cuda.empty_cache()
                else:
                    output = 'GPU {}: Epoch {}, Time: {:.0f} s, Batch {}/{}, lr: {:.6f}, train-loss: {:.3f} '.format(gpu_id, epoch + resume_epoch, time.time() - start_time, i + 1, len(training_dataloader), current_learning_rate, training_loss / settings.print_interval)
                print(output)
                logfile.write(output + '\n')
                training_loss = 0

        # Learning rate update:
        update_flag = torch.zeros(1).to(Settings().computing.device)
        if(Settings().computing.local_process_rank == 0):  # Decision made based on the loss of the first process
            prev_loss = net_module.training_loss.item()
            current_loss = np.asarray(losses).mean()
            room_for_improvement = max(Settings().network.min_room_for_improvement, prev_loss - Settings().network.target_loss)
            loss_change = (prev_loss - current_loss) / room_for_improvement
            print('GPU {}: Average training loss reduced {:.2f}% from the previous epoch.'.format(gpu_id, loss_change*100))
            if loss_change < Settings().network.min_loss_change_ratio:
                update_flag[0] = 1
                print('GPU {}: Because loss change {:.2f}% <= {:.2f}%, the learning rate is halved: {} --> {}'.format(gpu_id, loss_change*100, Settings().network.min_loss_change_ratio*100, optimizer.param_groups[0]['lr'] * 2, optimizer.param_groups[0]['lr']))
                print('GPU {}: Consecutive LR updates: {}'.format(gpu_id, net_module.consecutive_lr_updates[0]))
            else:
                net_module.consecutive_lr_updates[0] = 0

            net_module.training_loss[0] = current_loss

        if(Settings().computing.world_size > 1):  # Broadcast decision to all processes
            torch.distributed.broadcast(update_flag, 0)

        if(update_flag[0] == 1):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/2
            net_module.consecutive_lr_updates[0] += 1    
    
        if(Settings().computing.local_process_rank == 0): 
            network_io.save_state(output_filename, epoch + resume_epoch, net, optimizer)

        if net_module.consecutive_lr_updates[0] >= Settings().network.max_consecutive_lr_updates:
            print('GPU {}: Stopping training because reached {} consecutive LR updates!'.format(gpu_id, Settings().network.max_consecutive_lr_updates))
            return net_module, True, epoch + resume_epoch

    logfile.close()

    return net_module, False, epoch + resume_epoch



def _shuffle_data(segment_ids, speaker_labels, seed):
    random.seed(seed)
    c = list(zip(segment_ids, speaker_labels))
    random.shuffle(c)
    return zip(*c)

def _split_to_train_and_validation(data, n_validation, seed=101):
    np.random.seed(seed)
    n_utterances = len(data.utterances)
    validation_indices = np.random.choice(n_utterances, n_validation, replace=False)
    training_indices = np.setdiff1d(np.arange(n_utterances), validation_indices)
    n_validation = validation_indices.size
    n_training = training_indices.size
    print('Training set of {} utterances divided randomly to sets of {} and {} for training and validation.'.format(n_utterances, n_training, n_validation))
    training_data = UtteranceList([data[i] for i in np.nditer(training_indices)], name='training')
    validation_data = UtteranceList([data[i] for i in np.nditer(validation_indices)], name='validation')
    return training_data, validation_data

def _select_random_subset(data, n):
    print('Selecting random subset of training data for accuracy computation...')
    indices = np.random.choice(len(data.utterances), n, replace=False)
    subset_data = UtteranceList([data[i] for i in np.nditer(indices)], name='training_subset')
    return subset_data

def _init_optimizer(net, settings):
    params = get_weight_decay_param_groups(net, settings.weight_decay_skiplist)
    if settings.optimizer == 'sgd':
        return optim.SGD(params, lr=settings.initial_learning_rate, weight_decay=settings.weight_decay, momentum=settings.momentum)
    sys.exit('Unsupported optimizer: {}'.format(settings.optimizer))

def get_weight_decay_param_groups(model, skip_list):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if builtins.any(x in name for x in skip_list):
            no_decay.append(param)
            print('No weight decay applied to {}'.format(name))
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': Settings().network.weight_decay}]

def print_learnable_parameters(model: torch.nn.Module):
    print('Learnable parameters of the model:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

def print_once(string: str):
    if(Settings().computing.local_process_rank == 0):
        print(str)


