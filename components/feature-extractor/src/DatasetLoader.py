#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
from scipy.io import wavfile
from queue import Queue

def round_down(num, divisor):
    return num - (num%divisor)

# @brief read WAV file and convert to torch tensor
# @credit clovaai/voxceleb_trainer
def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # desired length of audio segment
    desired_audio_length = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)

    # actual length of audio segment
    actual_audio_length = audio.shape[0]

    # zero-pad audio segment if it's smaller than desired
    if actual_audio_length <= desired_audio_length:
        shortage    = math.floor( ( desired_audio_length - actual_audio_length + 1 ) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        actual_audio_length   = audio.shape[0]

    # grab 'desired_audio_length'-long subset of audio segment
    feats = []
    if evalmode and max_frames != 0:
        startframe = numpy.linspace(0,actual_audio_length-desired_audio_length,num=num_eval)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+desired_audio_length])
    else:
        feats.append(audio)

    # load into torch float tensor
    feat = numpy.stack(feats,axis=0)
    feat = torch.FloatTensor(feat)

    return feat;

class DatasetLoader(object):
    def __init__(self, dataset_file_name, batch_size, max_frames, max_seg_per_spk, nDataLoaderThread, gSize, new_train_path, maxQueueSize = 10, **kwargs):
        self.dataset_file_name = dataset_file_name;
        self.nWorkers = nDataLoaderThread;
        self.max_frames = max_frames;
        self.max_seg_per_spk = max_seg_per_spk;
        self.batch_size = batch_size;
        self.maxQueueSize = maxQueueSize;

        self.data_dict = {};
        self.data_list = [];
        self.nFiles = 0;
        self.gSize  = gSize; ## number of clips per sample (e.g. 1 for softmax, 2 for triplet or pm)

        self.dataLoaders = [];
        self.utterances = []
        self.num_utterances = 0

        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            while True:
                line = dataset_file.readline();
                if not line:
                    break;

                data = line.split();
                speaker_name = data[0];
                filename = os.path.join(new_train_path,data[1]);
                self.utterances.append({'speaker_name': speaker_name, 'filename': filename})
                self.num_utterances += 1

        ### Initialize Workers...
        self.datasetQueue = Queue(self.maxQueueSize);

    # @TODO load in batches for performance boost
    def dataLoaderThread(self, thread_index):
        iter_index = 0
        done = False
        while(not done):
            # wait if queue is full
            if(self.datasetQueue.full() == True):
                time.sleep(1.0);
                continue;

            # get index of beginning of batch
            inter_thread_group_offset = iter_index * self.nWorkers * self.batch_size
            intra_thread_group_offset = thread_index * self.batch_size
            batch_start_index = inter_thread_group_offset + intra_thread_group_offset

            batch_utterance_feats = []
            batch_utterance_paths = []

            # add utterance batch to data queue
            for utterance_index in range(batch_start_index, batch_start_index + self.batch_size):
                # exit if this loader has finished all its utterances
                if utterance_index >= self.num_utterances:
                    done = True
                    break

                utterance_path = self.utterances[utterance_index]['filename']

                # append their features, paths, and labels
                batch_utterance_paths.append(utterance_path)
                batch_utterance_feats.append(loadWAV(utterance_path, self.max_frames, evalmode=False))

            # push it on
            self.datasetQueue.put([batch_utterance_feats, batch_utterance_paths]);

            iter_index += 1

        print(f"data loader #{thread_index} finished")

    def __iter__(self):
        # start data loader threads
        for index in range(0, self.nWorkers):
            self.dataLoaders.append(threading.Thread(target = self.dataLoaderThread, args = [index]));
            self.dataLoaders[-1].start();

        return self;

    # @TODO refactor
    def __next__(self):

        while(True):
            isFinished = True;

            # return the first element in the datasetQueue if non-empty
            if(self.datasetQueue.empty() == False):
                return self.datasetQueue.get();

            # set isFinished = False if any data loaders are still alive
            for index in range(0, self.nWorkers):
                if(self.dataLoaders[index].is_alive() == True):
                    isFinished = False;
                    break;

            # if isFinished == False, give workers time to work, and skip to the
            # next iteration
            if(isFinished == False):
                time.sleep(1.0);
                continue;

            # if isFinished == True, join against all workers and cleanup
            for index in range(0, self.nWorkers):
                self.dataLoaders[index].join();

            self.dataLoaders = [];
            raise StopIteration;

    def __len__(self):
        return math.ceil(self.num_utterances / self.batch_size)

    def __call__(self):
        pass;

    def getDatasetName(self):
        return self.dataset_file_name;

    def qsize(self):
        return self.datasetQueue.qsize();
