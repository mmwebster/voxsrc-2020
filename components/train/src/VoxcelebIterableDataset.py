#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import random
import os
import threading
import time
import queue
from utils.misc_utils import print_throttler

# @credit github.com/clovaai/voxceleb_trainer
def round_down(num, divisor):
    return num - (num%divisor)

# @brief Zero pad spectrogram in frequency domain if its length is below padded_length
# @TODO How will the network respond to padded spectrograms? (original
#       implementation padded the time-domain audio)
def pad_spectrogram(spectrogram, min_frames):
    actual_frames = spectrogram.shape[2]
    if actual_frames < min_frames:
        num_frequencies = spectrogram.shape[1]
        # create zeros of desired shape
        padded_spectrogram = np.zeros((1, min_frames, num_frequencies))
        # insert original data and return
        padded_spectrogram[:, :spectrogram.shape[1],:spectrogram.shape[2]] = spectrogram
        return padded_spectrogram
    else:
        return spectrogram

# @brief extract a random or deterministic subset of contiguous frames from a
#        spectrogram as a numpy array
# @param spectrogram 3-dim numpy array containing spectrogram. 2nd axis is
#        frequency, 3rd axis it time, 1st axis is a trivial 1-long list
# @note Not sure why the trivial 1st dimension is necessary but training code
#       expects this 3-dim spectrogram
# @note desired_frames was previously named "max_frames", as somewhat of a
#       misnomer. Utterances with insufficient length would be zero-padding in
#       time domain, but it still was the actual number of spectrogram frames
#       always returned, not a max for a range
def extract_subset_of_spectrogram(spectrogram, desired_frames):
    # pad if too small for network dims
    padded_spectrogram = pad_spectrogram(spectrogram, desired_frames)

    # select a random start frame of subset
    actual_frames = spectrogram.shape[2]
    start_frame = np.int64(random.random()*(actual_frames-desired_frames))

    # return 'desired_frames'-long subset of audio segment
    return padded_spectrogram[:,:,int(start_frame):int(start_frame)+desired_frames]

# @TODO Change input from std dev to dB, after figuring out power when
#       adding random noise to a log mel spectrogram
def add_gaussian_noise_to_spectrogram(spectrogram, noise_std_dev):
    mean = 0
    noise = np.random.normal(mean, noise_std_dev, spectrogram.shape)
    return spectrogram + noise

# @brief Extracts n_subsets of overlapping desired_frames length. Each subset's offset
#        from the previous frame is dependent on its total length and the number
#        of subsets
def extract_eval_subsets_from_spectrogram(spectrogram, desired_frames, n_subsets = 10):
    # pad if too small for network dims
    padded_spectrogram = pad_spectrogram(spectrogram, desired_frames)

    # make list of overlapping n_subsets start frames spanning the entire utterance
    actual_frames = spectrogram.shape[2]
    start_frames = np.linspace(0, actual_frames - desired_frames, n_subsets)

    # append each utterance subset
    utterance_subsets = []
    for start_frame in start_frames:
        # @note appending the first element in first axis in order to get a list
        #       of NxM tensors, rather than 1xNxM
        utterance_subsets.append(padded_spectrogram[0,:,
            int(start_frame):int(start_frame)+desired_frames])

    # return stacked 3d tensor instead of list of 3d tensors
    return np.stack(utterance_subsets, axis=0)


# @brief Iteratable for Voxceleb 2 dataset, constructed from a training list
#        file, following the PyTorch IterableDataset interface. Meant for
#        parallelization with PyTorch's DataLoader
# @note Must set "batch_size=None" in the consuming DataLoader in order to allow
#       this class to manually compose batches. This is a quick fix to maintain
#       compatibility with VoxSRC-provided baseline dataloading code
class VoxcelebIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_file_name, batch_size, max_frames,
            max_seg_per_spk, n_data_loader_thread,
            gSize, new_train_path, maxQueueSize = 50, gaussian_noise_std = .9,
            **kwargs):
        super(VoxcelebIterableDataset).__init__()
        self.n_speakers = gSize
        self.batch_size = batch_size

        # legacy baseline params (to refactor)
        self.dataset_file_name = dataset_file_name
        self.new_train_path = new_train_path
        self.max_frames = max_frames
        self.max_seg_per_spk = max_seg_per_spk
        self.gaussian_noise_std = gaussian_noise_std

        self.data_dict = {}
        self.data_list = []
        self.nFiles = 0
        # number of clips per sample (e.g. 1 for softmax, 2 for triplet or pm)
        self.gSize  = gSize

        # populate data_dict with utterance paths and speaker IDs
        self.data_dict = self.legacy_utterance_path_read(self.dataset_file_name,
                self.new_train_path)

        # misc preprocessing to prevent duplicate speaker IDs in batches
        self.data_list, self.data_label, self.nFiles = self.legacy_batch_prep(
                self.data_dict, self.max_seg_per_spk, self.gSize,
                self.batch_size)

    # @brief utterance file path reading
    # @credit github.com/clovaai/voxceleb_trainer
    def legacy_utterance_path_read(self, utterance_list_file_path,
            utterance_data_root_path):
        data_dict = {}
        with open(utterance_list_file_path) as dataset_file:
            while True:
                line = dataset_file.readline();
                if not line:
                    break;

                data = line.split();
                speaker_name = data[0];
                filename = os.path.join(utterance_data_root_path, data[1])

                if not (speaker_name in data_dict):
                    data_dict[speaker_name] = [];

                data_dict[speaker_name].append(filename);

            return data_dict

    # @brief Misc preperation for batching in workers w/o duplicate speaker IDs
    #        per batch 
    # @credit github.com/clovaai/voxceleb_trainer
    def legacy_batch_prep(self, data_dict, max_seg_per_spk, gSize, batch_size):
        dictkeys = list(data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = data_dict[key]
            numSeg  = round_down(min(len(data),max_seg_per_spk),gSize)

            rp      = lol(np.random.permutation(len(data))[:numSeg],gSize)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid           = np.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        data_list  = [flattened_list[i] for i in mixmap]
        data_label = [flattened_label[i] for i in mixmap]

        ## Iteration size
        nFiles = len(data_label);

        return data_list, data_label, nFiles

    def __len__(self):
        return self.nFiles

    def getDatasetName(self):
        return self.dataset_file_name;

    def next_batch_exists(self, batch_index):
        return batch_index + self.batch_size <= self.nFiles

    # @brief Common data loading for all workers
    # @return Iterator for a worker's subset of the dataset. This iterator
    #         yields full batches of
    #         n_speakers x batch_size x frequency_bins x utterance_length
    # @note Corresponding data loader must have "batch_size = None" to indicate
    #       that the dataset is performing batching on its own
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        print(f"VoxcelebIterableDataset: Starting worker thread #{worker_info.id}")
        index = worker_info.id * self.batch_size;

        if (index >= self.nFiles):
            print(f"VoxcelebIterableDataset: ERROR -> Invalid data loader #{worker_info.id}")
            return

        while self.next_batch_exists(index):
            in_data = [];
            for ii in range(0,self.gSize):
                feat = []
                for ij in range(index,index+self.batch_size):
                    # @note if there aren't enough independent
                    #       speakers, the batch won't be fillable
                    # @note casting pre-extracted float16 features to float32 so
                    #       as to not inadvertently introduce network quantization
                    utterance_file_path = self.data_list[ij][ii].replace(".wav", ".npy")
                    full_spectrogram = np.load(utterance_file_path)
                    subset_spectrogram = extract_subset_of_spectrogram(full_spectrogram, self.max_frames)
                    noisy_spectrogram = add_gaussian_noise_to_spectrogram(
                            subset_spectrogram, noise_std_dev=self.gaussian_noise_std)
                    feat.append(torch.FloatTensor(noisy_spectrogram));
                in_data.append(torch.cat(feat, dim=0));

            in_label = np.asarray(self.data_label[index:index+self.batch_size]);

            # send the batch to the Dataloader, iterable style
            yield [in_data, in_label]

            index += self.batch_size * worker_info.num_workers;

        print(f"VoxcelebIterableDataset: Stopping worker #{worker_info.id}")
