# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import Tuple, Type, Union
import time

import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

from asvtorch.src.misc.miscutils import test_finiteness

from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader
from asvtorch.src.settings.settings import Settings
from asvtorch.src.ivector.sufficient_stats import SufficientStats


class TestDataset(Dataset):
    def __init__(self, data: UtteranceList, usage='validation'):
        assert usage in ['extraction', 'validation']
        self.usage = usage
        self.batch_data = []
        n_segments = len(data.utterances)
        data.utterances.sort(key=lambda x: x.get_minimum_selected_frame_count()) # Sorting based on utterance length
        if usage == 'extraction':
            max_cut_portion = Settings().network.max_test_cut_portion
        else:
            max_cut_portion = Settings().network.max_val_cut_portion
        batch_frames_sum = 0
        batch_size_sum = 0
        index = 0
        while index < n_segments:
            fixed_segment_length = data.utterances[index].get_minimum_selected_frame_count()
            if fixed_segment_length > Settings().network.max_batch_size_in_frames:
                fixed_segment_length = Settings().network.max_batch_size_in_frames
            max_segment_length = fixed_segment_length / (1 - max_cut_portion)
            frames_filled = 0
            batch = []
            while frames_filled + fixed_segment_length <= Settings().network.max_batch_size_in_frames:
                frames_filled += fixed_segment_length
                batch.append(data.utterances[index])
                index += 1
                if index == n_segments or data.utterances[index].get_minimum_selected_frame_count() > max_segment_length:
                    break
            batch_frames_sum += frames_filled
            batch_size_sum += len(batch)
            self.batch_data.append((batch, fixed_segment_length))
        print('{} Testing minibatches created!'.format(len(self.batch_data)))
        print('  - Maximum portion of cutted speech (setting): {} %'.format(max_cut_portion * 100))
        print('  - Maximum batch size in frames (setting): {}'.format(Settings().network.max_batch_size_in_frames))
        print('  - Average batch size in frames (realized): {:.1f}'.format(batch_frames_sum / len(self.batch_data)))
        print('  - Average batch size in utterances (realized): {:.1f}'.format(batch_size_sum / len(self.batch_data)))

    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.LongTensor]]:
        feat_list = []
        batch, fixed_segment_length = self.batch_data[index]
        if self.usage == 'extraction':
            for utt in batch:
                feats = FeatureLoader().load_features(utt)
                feat_list.append(feats[:fixed_segment_length, :])
            features = torch.from_numpy(np.transpose(np.dstack(feat_list), (2, 1, 0)))
            return features
        labels = []
        for utt in batch:
            feats = FeatureLoader().load_features(utt)
            feat_list.append(feats[:fixed_segment_length, :])
            labels.append(utt.spk_id)
            features = torch.from_numpy(np.transpose(np.dstack(feat_list), (2, 1, 0)))
        return features, torch.LongTensor(labels)

def compute_losses_and_accuracies(dataloaders, network, loss_function):
    network.eval()
    losses_and_accuracies = []
    for dataloader in dataloaders:
        correct_classifications = 0
        segment_count = 0
        loss_sum = 0
        for batch in dataloader:
            features, speaker_labels = batch
            features = features.to(Settings().computing.device)
            batch_size = speaker_labels.numel()
            speaker_labels = speaker_labels.to(Settings().computing.device)
            with torch.no_grad():
                outputs = network(features)
                loss = loss_function(outputs, speaker_labels)
            loss_sum += loss.item() * batch_size
            outputs = torch.argmax(outputs, dim=1)
            correct_classifications += torch.sum(outputs == speaker_labels)
            segment_count += batch_size
        losses_and_accuracies.append((loss_sum/segment_count, float(correct_classifications) / segment_count * 100))
    network.train()
    return losses_and_accuracies

# The input utterance list will get sorted (in-place) according to utterance length, returned embeddings are in the sorted order
def extract_embeddings(data: UtteranceList, network: Type[torch.nn.Module]):
    print('Extracting {} embeddings...'.format(len(data.utterances)))
    network.eval()
    dataloader = get_dataloader(data, usage='extraction')
    embeddings = torch.zeros(len(data.utterances), Settings().network.embedding_size)
    counter = 0
    start_time = time.time()
    for index, batch in enumerate(dataloader):
        features = batch
        features = features.to(Settings().computing.device)
        with torch.no_grad():
            batch_embeddings = network(features, 'extract_embeddings')
        embeddings[counter:counter+batch_embeddings.size()[0], :] = batch_embeddings
        test_finiteness(batch_embeddings, str(index))
        counter += batch_embeddings.size()[0]
        if index % (Settings().network.extraction_print_interval) == (Settings().network.extraction_print_interval) - 1:
            print('{:.0f} seconds elapsed, {}/{} batches, {}/{} utterances'.format(time.time() - start_time, index+1, len(dataloader), counter, len(data.utterances)))
    data.embeddings = embeddings

# The input utterance list will get sorted (in-place) according to utterance length, returned embeddings are in the sorted order
def extract_neural_features(data: UtteranceList, network: Type[torch.nn.Module]):
    print('Extracting neural features for {} utterances...'.format(len(data.utterances)))
    network.eval()
    dataloader = get_dataloader(data, usage='extraction')
    feature_list = []
    counter = 0
    start_time = time.time()
    for index, batch in enumerate(dataloader):
        features = batch
        features = features.to(Settings().computing.device)
        with torch.no_grad():
            batch_features = network(features, 'extract_features').transpose(1, 2)
        feature_list.extend([x.squeeze(0) for x in torch.split(batch_features, 1, dim=0)])
        counter += len(batch_features)
        if index % (Settings().network.extraction_print_interval) == (Settings().network.extraction_print_interval) - 1:
            print('{:.0f} seconds elapsed, {}/{} batches, {}/{} utterances'.format(time.time() - start_time, index+1, len(dataloader), counter, len(data.utterances)))
    data.neural_features = feature_list

# The input utterance list will get sorted (in-place) according to utterance length, returned embeddings are in the sorted order
def extract_stats(data: UtteranceList, network: Type[torch.nn.Module], second_order=True):
    print('Extracting sufficient statistics for {} utterances...'.format(len(data.utterances)))

    network.eval()
    dataloader = get_dataloader(data, usage='extraction')
    feat_dim = Settings().network.stat_size
    zeroth = torch.zeros(len(data.utterances), Settings().network.n_clusters)
    first = torch.zeros(len(data.utterances), Settings().network.n_clusters, feat_dim)
    if second_order:
        mode = 'extract_training_stats'
        second_sum = torch.zeros(Settings().network.n_clusters, feat_dim, feat_dim)
    else:
        mode = 'extract_testing_stats'  
    counter = 0
    start_time = time.time()
    for index, batch in enumerate(dataloader):
        features = batch
        features = features.to(Settings().computing.device)
        with torch.no_grad():
            stats = network(features, mode)
        z = stats[0].cpu()
        f = stats[1].cpu()
        zeroth[counter:counter+stats[0].size()[0], :] = z
        first[counter:counter+stats[1].size()[0], :, :] = f
        if second_order:
            second_sum += stats[2].cpu()
            test_finiteness(second_sum, 'second ' + str(index))
        test_finiteness(z, 'zeroth ' + str(index))
        test_finiteness(f, 'first ' + str(index))
        counter += stats[0].size()[0]
        if index % (Settings().network.extraction_print_interval) == (Settings().network.extraction_print_interval) - 1:
            print('{:.0f} seconds elapsed, {}/{} batches, {}/{} utterances'.format(time.time() - start_time, index+1, len(dataloader), counter, len(data.utterances)))
    if not second_order:
        second_sum = None
    data.stats = SufficientStats(zeroth, first, second_sum)

def _collater(batch):
    # Batch is already formed in the DataSet object (batch consists of a single element, which is actually the batch itself).
    return batch[0]

def get_dataloader(data, usage='validation'):
    dataset = TestDataset(data, usage)
    print('Feature loader for {} initialized!'.format(usage))
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=Settings().computing.network_dataloader_workers, collate_fn=_collater)
