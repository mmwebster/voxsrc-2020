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
from queue import Queue

def round_down(num, divisor):
    return num - (num%divisor)

# @brief extract a random subset of contiguous frames from a spectrogram as a
#        numpy array
# @param spectrogram 3-dim numpy array containing spectrogram. 2nd axis is
#        frequency, 3rd axis it time, 1st axis is a trivial 1-long list
# @note Not sure why the trivial 1st dimension is necessary but training code
#       expects this 3-dim spectrogram
def extract_subset_of_spectrogram(spectrogram, desired_frames):
    # actual length of audio segment
    actual_frames = spectrogram.shape[2]
    #print(f"spectrogram dims are {spectrogram.shape}")

    padded_spectrogram = spectrogram
    # zero-pad audio segment if it's smaller than desired
    # @TODO How will the network respond to padded spectrograms? (original
    #       source padded the time-domain audio)
    if actual_frames < desired_frames:
        num_frequencies = spectrogram.shape[1]
        # create zeros of desired shape
        padded_spectrogram = np.zeros((1, desired_frames, num_frequencies))
        # insert original data
        padded_spectrogram[:, :spectrogram.shape[1],:spectrogram.shape[2]] = spectrogram

    # select a random start frames of subset
    start_frame = np.int64(random.random()*(actual_frames-desired_frames))
    #print(f"start frame is {start_frame}, ")

    # return 'desired_frames'-long subset of audio segment
    return padded_spectrogram[:,:,int(start_frame):int(start_frame)+desired_frames]


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
        audio       = np.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        actual_audio_length   = audio.shape[0]

    # set deterministic or random initial frame based on eval/not eval
    if evalmode:
        startframe = np.linspace(0,actual_audio_length-desired_audio_length,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(actual_audio_length-desired_audio_length))])

    # grab 'desired_audio_length'-long subset of audio segment
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        # @TODO Why is startframe an array? Currently iterating through 1 element...
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+desired_audio_length])

    # load into torch float tensor
    feat = np.stack(feats,axis=0)
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
            for ii in range(0,self.gSize):
                feat = []
                for ij in range(index,index+self.batch_size):
                    # @note if there aren't enough independent
                    #       speakers, the batch won't be fillable
                    # @note casting pre-extracted float16 features to float32 so
                    #       as to not inadvertently introduce network quantization
                    full_spectrogram = np.load(self.data_list[ij][ii].replace(".wav", ".npy")).astype('float32')
                    subset_spectrogram = extract_subset_of_spectrogram(full_spectrogram, self.max_frames)
                    feat.append(torch.FloatTensor(subset_spectrogram));
                in_data.append(torch.cat(feat, dim=0));

            in_label = np.asarray(self.data_label[index:index+self.batch_size]);
            
            self.datasetQueue.put([in_data, in_label]);

            index += self.batch_size*self.nWorkers;

            if(index+self.batch_size > self.nFiles):
                break;



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
        for index in range(0, self.nWorkers):
            self.dataLoaders.append(threading.Thread(target = self.dataLoaderThread, args = [index]));
            self.dataLoaders[-1].start();

        return self;


    def __next__(self):

        while(True):
            isFinished = True;
            
            if(self.datasetQueue.empty() == False):
                return self.datasetQueue.get();
            for index in range(0, self.nWorkers):
                if(self.dataLoaders[index].is_alive() == True):
                    isFinished = False;
                    break;

            if(isFinished == False):
                time.sleep(1.0);
                continue;


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
