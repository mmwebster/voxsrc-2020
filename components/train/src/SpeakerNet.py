#!/usr/bin/python
#-*- coding: utf-8 -*-

import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.cuda.amp as amp
import math, pdb, sys, random
import torch.nn.functional as F
from loss.angleproto import AngleProtoLoss
from utils.misc_utils import print_throttler
import time, os, itertools, shutil, importlib
from IterableEvalDataset import IterableEvalDataset
from baseline_misc.tuneThreshold import tuneThresholdfromScore

class SpeakerNet(nn.Module):

    def __init__(self, device, max_frames, batch_size, eval_batch_size,
            n_data_loader_thread, lr = 0.0001, margin = 1, scale = 1,
            hard_rank = 0, hard_prob = 0, model="alexnet50", nOut = 512,
            nSpeakers = 1000, optimizer = 'adam', encoder_type = 'SAP',
            normalize = True, trainfunc='contrastive', **kwargs):
        super(SpeakerNet, self).__init__();

        argsdict = {'nOut': nOut, 'encoder_type':encoder_type}

        self.device = device
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.n_data_loader_thread = n_data_loader_thread

        # grab actual model version
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__(model)

        # set model as __S__ member
        self.__S__ = SpeakerNetModel(**argsdict).to(self.device);

        # remove variable loss from baseline, only using angular prototypical loss
        self.__L__ = AngleProtoLoss(self.device).to(self.device)
        self.__train_normalize__    = True
        self.__test_normalize__     = True

        if optimizer == 'adam':
            self.__optimizer__ = torch.optim.Adam(self.parameters(), lr = lr);
        elif optimizer == 'sgd':
            self.__optimizer__ = torch.optim.SGD(self.parameters(), lr = lr,
                    momentum = 0.9, weight_decay=5e-5);
        else:
            raise ValueError('Undefined optimizer.')

        self.__max_frames__ = max_frames;

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    # @TODO figure out how to get length of dataset through DataLoader while
    #       passing a batch_size=None to the DataLoader
    def train_on(self, loader, data_length):

        self.train();

        stepsize = self.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy

        criterion = torch.nn.CrossEntropyLoss()
        # mixed precision scaler
        scaler = amp.GradScaler()

        print_interval_percent = 2
        print_interval = 0
        print_interval_start_time = time.time()
        epoch_start_time = time.time()


        total_elapsed_dequeue_time = 0
        batch_dequeue_start_time = time.time()
        for data, data_label in loader:
            total_elapsed_dequeue_time += time.time() - batch_dequeue_start_time
            # init print interval after data loader has set its length during __iter__
            if print_interval == 0:
                num_batches = (data_length/self.batch_size)
                print_interval = max(int(num_batches*print_interval_percent/100), 1)
                print(f"SpeakerNet: Starting training @ {print_interval_percent}%"
                      f" update interval")

            self.zero_grad();

            feat = []
            # use autocast for half precision where possible
            with amp.autocast():
                # @note The 'data' returned by the loader is n_speakers * batch_dims
                for inp in data:
                    outp      = self.__S__.forward(inp.to(self.device))
                    if self.__train_normalize__:
                        outp   = F.normalize(outp, p=2, dim=1)
                    feat.append(outp)

                feat = torch.stack(feat,dim=1).squeeze()
                label   = torch.LongTensor(data_label).to(self.device)
                nloss, prec1 = self.__L__.forward(feat,label)

                loss    += nloss.detach().cpu();
                top1    += prec1
                counter += 1;
                index   += stepsize;

            # run backward pass and step optimizer using the autoscaler
            # to mitigate half-precision convergence issues
            scaler.scale(nloss).backward()
            scaler.step(self.__optimizer__)
            scaler.update()

            if counter % print_interval == 0:
                # not sure how to format in f-format str
                interval_elapsed_time = time.time() - print_interval_start_time
                print_interval_start_time = time.time()
                eer_str = "%2.3f%%"%(top1/counter)
                # misc progress updates and estimates
                progress_percent = int(index * 100 / data_length)
                num_samples_processed = print_interval * self.batch_size
                sample_train_rate = num_samples_processed / interval_elapsed_time
                epoch_train_period = (data_length / sample_train_rate) / 60
                print(f"SpeakerNet: Processed {progress_percent}% => "
                      f"Loss {loss/counter:.2f}, "
                      f"EER/T1 {eer_str}, "
                      f"Train rate {epoch_train_period:.2f} mins/epoch, "
                      f"Total batch fetch time {total_elapsed_dequeue_time:.2f} (s)")
            batch_dequeue_start_time = time.time()

        print(f"SpeakerNet: Finished epoch in {(time.time() - epoch_start_time)/60:.2f} mins")
        return (loss/counter, top1/counter);

    # @brief Evaluate the model on the provided test list and test data
    # @param test_list_path Full path to file containing test pairs with
    #                       lines [label, utterance_relative_path_a,
    #                             utterance_relative_path_b]
    # @param test_data_path Full path to directory containing all test data
    # @param num_utterance_eval_subsets The number of overlapping fixed-length
    #                                   spectrogram subsets to extract from the
    #                                   test utterances. With 10, for example,
    #                                   10 fixed-length spectrograms will be
    #                                   extracted from a single utterance, with
    #                                   a time offset of (1/10)*utterance_length
    # @note DO NOT MODIFY THIS EVAL STRATEGY. The procedure for this EER calc is
    #       specified by VoxSRC competition and necessary for relative comparison
    #       of model performance
    # @TODO accept a data loader for validation data, just like the interface
    #       for training
    def evaluate_on(self, test_list_path, test_data_path,
            num_utterance_eval_subsets=10):

        self.eval();
        scores = []
        labels = []

        # create torch dataset and data loader for evaluation data
        iterable_eval_dataset = IterableEvalDataset(
                test_list_path=test_list_path,
                test_data_path=test_data_path,
                num_desired_frames=self.__max_frames__,
                num_utterance_eval_subsets=num_utterance_eval_subsets,
                batch_size=self.eval_batch_size)
        # @note num_workers is 1 greater than that for train, since eval's data loading 
        eval_loader = torch.utils.data.DataLoader(iterable_eval_dataset,
                batch_size = None, num_workers=self.n_data_loader_thread)

        print(f"SpeakerNet: Starting model eval on {len(eval_loader)} batches "
              f"of size {self.eval_batch_size}")

        # map of [path]->[embedding] to contain NxM features where N is the
        # number of utterance subsets extracted, and M is the dimension of the
        # model's embedding
        utterance_embedding_cache = {}

        start_time = time.time()

        for feature_utterance_path_lookup, spectrograms in eval_loader:
            # calculate embeddings for each utterance-subset spectrogram in tensor
            with torch.no_grad():
                embeddings = self.__S__.forward(spectrograms.to(self.device)).detach().cpu().numpy()

            # cache all the computed features
            for utterance_index, utterance_path in enumerate(feature_utterance_path_lookup):
                # compute subset of features tensor belonging to single utterance
                # (each utterance has multiple subsets extracted from its
                # spectrogram for evaluation)
                utterance_start_index = utterance_index * num_utterance_eval_subsets
                utterance_embedding_indices = range(utterance_start_index,
                        utterance_start_index + num_utterance_eval_subsets)
                # save the utterance embeddings
                utterance_embedding_cache[utterance_path] \
                        = embeddings[utterance_embedding_indices]
        print(f"SpeakerNet: Computed utterance segment embeddings "
              f"in {time.time() - start_time} (s)")
        start_time = time.time()

        # @TODO Parallelize compute-bound normalization and computation of scores on cpu
        # Read through test file and compute scores from test pair embedding
        # similarity
        with open(test_list_path) as utterance_test_pairs:
            line = utterance_test_pairs.readline()
            while line:
                data = line.split()
                label = data[0]
                ref_feat = torch.FloatTensor(utterance_embedding_cache[data[1]])
                com_feat = torch.FloatTensor(utterance_embedding_cache[data[2]])

                # optionally normalize
                if self.__test_normalize__:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                # compute and save score
                dist = F.pairwise_distance(
                            ref_feat.unsqueeze(-1).expand(-1,-1,
                                num_utterance_eval_subsets),
                            com_feat.unsqueeze(-1).expand(-1,-1,
                                num_utterance_eval_subsets).transpose(0,2)
                       ).detach().numpy()

                # append scores
                scores.append(-1 * np.mean(dist))
                labels.append(int(label))

                line = utterance_test_pairs.readline()

        print(f"SpeakerNet: Computed utterance test pair scores "
              f"in {time.time() - start_time} (s)")

        torch.cuda.empty_cache()

        return (scores, labels);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Update learning rate
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def updateLearningRate(self, alpha):

        learning_rate = []
        for param_group in self.__optimizer__.param_groups:
            param_group['lr'] = param_group['lr']*alpha
            learning_rate.append(param_group['lr'])

        return learning_rate;


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        torch.save(self.state_dict(), path);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save model
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveModel(self, path):
        torch.save(self, path);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);
