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

    # set deterministic or random initial frame based on eval/not eval
    if evalmode:
        startframe = numpy.linspace(0,actual_audio_length-desired_audio_length,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(actual_audio_length-desired_audio_length))])

    # grab 'desired_audio_length'-long subset of audio segment
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        # @TODO Why is startframe an array? Currently iterating through 1 element...
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+desired_audio_length])

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
        self.datasetQueue = Queue(self.maxQueueSize);
    

    def dataLoaderThread(self, nThreadIndex):
        
        index = nThreadIndex*self.batch_size;

        if(index >= self.nFiles):
            return;

        while(True):
            if(self.datasetQueue.full() == True):
                time.sleep(1.0);
                continue;

            in_data = [];
            multi_batch_paths = []
            # iterate through the 2nd dim of self.data_list, setting 'ii' to be
            # a position within a tuple
            for ii in range(0,self.gSize):
                batch_paths = []
                feat = []
                # iterate through the 1st dim of self.data_list, setting ij to
                # be a tuple index from "index" to "index + self.batch_size"
                for ij in range(index,index+self.batch_size):
                    # @warning if there aren't enough unique speakers, the batch
                    #          won't be fillable
                    # append a single utterance to the batch's features
                    feat.append(loadWAV(self.data_list[ij][ii], self.max_frames,
                        evalmode=False));
                    batch_paths.append(self.data_list[ij][ii])
                # append the batch containing unique speakers
                in_data.append(torch.cat(feat, dim=0));
                multi_batch_paths.append(batch_paths)

            in_label = numpy.asarray(self.data_label[index:index+self.batch_size]);

            self.datasetQueue.put([in_data, in_label, multi_batch_paths]);

            index += self.batch_size*self.nWorkers;

            if(index+self.batch_size > self.nFiles):
                break;



    def __iter__(self):

        dictkeys = list(self.data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        # contains list of speakers, each containing a list of file paths to
        # their utterances
        flattened_list = []
        # contains list of training labels (integer speaker IDs) where the index
        # corresponds to the respective place in the flattened_list (if it were
        # truly flattened, instead of a list of lists)
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            # key is speaker ID
            data    = self.data_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.gSize)

            rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.gSize)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        # contains utterance indices that will be used (after multi pair removal)
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        # contains 'nSpeakers' tuples of each speaker's utterances, multiples
        # tuples per speaker. Not sure how this is, when the code above "prevents
        # two pairs of the same speaker in the same batch"
        self.data_list  = [flattened_list[i] for i in mixmap]
        # contains integer speaker ID label for the respective tuple in
        # self.data_list
        self.data_label = [flattened_label[i] for i in mixmap]

        ## Iteration size
        self.nFiles = len(self.data_label);

        ### Make and Execute Threads...
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


    def __call__(self):
        pass;

    def getDatasetName(self):
        return self.dataset_file_name;

    def qsize(self):
        return self.datasetQueue.qsize();
