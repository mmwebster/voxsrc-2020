#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import random
import pdb
import os
import threading
import time
import math
from scipy.io import wavfile
import queue

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

class DatasetLoader(object):
    def __init__(self, dataset_file_name, batch_size, max_frames,
            max_seg_per_spk, n_data_loader_thread, gSize, new_train_path,
            maxQueueSize = 50, **kwargs):
        self.dataset_file_name = dataset_file_name;
        self.n_workers = n_data_loader_thread;
        self.max_frames = max_frames;
        self.max_seg_per_spk = max_seg_per_spk;
        self.batch_size = batch_size;
        self.maxQueueSize = maxQueueSize;

        self.data_dict = {};
        self.data_list = [];
        self.nFiles = 0;
        self.gSize  = gSize; ## number of clips per sample (e.g. 1 for softmax, 2 for triplet or pm)

        self.dataLoaders = [];

        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            while True:
                line = dataset_file.readline();
                if not line:
                    break;

                data = line.split();
                speaker_name = data[0];
                filename = os.path.join(new_train_path,data[1]);

                if not (speaker_name in self.data_dict):
                    self.data_dict[speaker_name] = [];

                self.data_dict[speaker_name].append(filename);

        ### Initialize Workers...
        self.datasetQueue = queue.Queue(self.maxQueueSize);

    def __len__(self):
        return self.nFiles

    def next_batch_exists(self, batch_index):
        return batch_index+self.batch_size <= self.nFiles

    def data_loader_thread(self, nThreadIndex):

        print(f"DatasetLoader: Starting worker thread #{nThreadIndex}")
        index = nThreadIndex*self.batch_size;

        if(index >= self.nFiles):
            return;

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
                    feat.append(torch.FloatTensor(subset_spectrogram));
                in_data.append(torch.cat(feat, dim=0));

            in_label = np.asarray(self.data_label[index:index+self.batch_size]);

            # try to enqueue batch until there's space
            failed_to_queue = True
            while failed_to_queue:
                try:
                    self.datasetQueue.put([in_data, in_label], timeout=.1)
                    failed_to_queue = False
                except queue.Full:
                    continue

            index += self.batch_size*self.n_workers;

        print(f"DatasetLoader: Stopping worker thread #{nThreadIndex}")

    def __iter__(self):

        dictkeys = list(self.data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = self.data_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.gSize)
            
            rp      = lol(np.random.permutation(len(data))[:numSeg],self.gSize)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid           = np.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        self.data_list  = [flattened_list[i] for i in mixmap]
        self.data_label = [flattened_label[i] for i in mixmap]
        
        ## Iteration size
        self.nFiles = len(self.data_label);

        ### Make and Execute Threads...
        for index in range(0, self.n_workers):
            self.dataLoaders.append(threading.Thread(target = self.data_loader_thread, args = [index]));
            self.dataLoaders[-1].start();

        return self;

    # @brief Returns true if all workers are dead, false otherwise
    def all_workers_dead(self):
        all_workers_dead = True
        for thread_index in range(self.n_workers):
            if self.dataLoaders[thread_index].is_alive():
                all_workers_dead = False
                break
        return all_workers_dead

    def join_workers(self):
        print(f"DatasetLoader: Joining against worker threads")
        for thread_index in range(self.n_workers):
            self.dataLoaders[thread_index].join()

    def __next__(self):
        done = False
        while not done:
            try:
                # grab a set of batches (might just be one batch, not sure)
                return self.datasetQueue.get(timeout=.1)
            except queue.Empty:
                print("DatasetLoader: Timed out on fetching batch set from "
                      "queue. Can the data loaders populate it fast enough?!")
                # if all data loaders are dead, then the epoch is done
                if self.all_workers_dead():
                    print("DatasetLoader: All workers are dead.")
                    done = True
        # join against data loaders once the epoch has completed
        self.join_workers()
        # clear data loaders, and stop the iterations
        self.dataLoaders = [];
        raise StopIteration;

    def __call__(self):
        pass;

    def getDatasetName(self):
        return self.dataset_file_name;

    def qsize(self):
        return self.datasetQueue.qsize();
