#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, os

# Add common src dir to python import path (varies between runs on and
# off the training cluster)
sys.path.insert(0, os.getenv('VOX_COMMON_SRC_DIR'))

import time, os, argparse, socket
import random
import math
import numpy
import pdb
import torch
import torchaudio
import glob
from baseline_misc.tuneThreshold import tuneThresholdfromScore
from DatasetLoader import DatasetLoader
import subprocess
import time
from pathlib import Path
from data_fetch import download_gcs_dataset, extract_gcs_dataset, \
                     transcode_gcs_dataset, set_loc_paths_from_gcs_dataset,\
                     download_blob, upload_blob
import yaml
import pwd
import google
import wandb
import matplotlib.pyplot as plt

# @TODO Strip this file of all training related stuff, not needed for a pure
#       feature extractor

# @brief Generate a unique run ID when not run in kubeflow (kubeflow passes
#        its own default run ID) in order to store training artifacts
#        for resume after preemption
# @note To manually resume a run outside of kubeflow, pass the run ID
#       printed in the run with the "--run-id" flag
def gen_run_id():
    user_id = pwd.getpwuid( os.getuid() )[ 0 ]
    wandb_id = wandb.util.generate_id()
    return f"{user_id}-{wandb_id}"

parser = argparse.ArgumentParser(description = "Feature Extractor");

## New args to support running on kubernetes/kubeflow
# @note "tmp" denotes that this output data will not be captured by
#       the kubeflow pipeline or made available to downstream components
# set --data-bucket in order to fetch lists and data from GCS before reading
# them from local filesystem

# temporary/internal outputs
parser.add_argument('--data-bucket', type=str)
parser.add_argument('--save-tmp-data-to', type=str, default="./tmp/data/")
parser.add_argument('--skip-data-fetch', action='store_true')
parser.add_argument('--reset-training', action='store_true', help='Reset \
        training to first epoch, regardless of previously saved model checkpoints')
parser.add_argument('--save-tmp-model-to', type=str, default="./tmp/model/");
parser.add_argument('--save-tmp-results-to', type=str, default="./tmp/results/");
parser.add_argument('--save-tmp-feats-to', type=str, default="./tmp/feats/");
parser.add_argument('--save-tmp-wandb-to', type=str, default="./tmp/");

parser.add_argument('--no-cuda', action='store_true');
parser.add_argument('--set-seed', action='store_true');

parser.add_argument('--checkpoint-bucket', type=str,
        default="voxsrc-2020-checkpoints-dev");
parser.add_argument('--run-id', type=str, default=f"{gen_run_id()}");

# permanent/component outputs
parser.add_argument('--save-model-to', type=str, default="./out/model.txt")

## Data loader
parser.add_argument('--max_frames', type=int, default=200,  help='Input length to the network');
parser.add_argument('--batch_size', type=int, default=200,  help='Batch size');
# ^^^ use --batch_size=30 for small datasets that can't fill an entire 200 speaker pair/triplet batch
parser.add_argument('--max_seg_per_spk', type=int, default=100, help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=8, help='Number of loader threads');

## Training details
# @TODO disentangle learning rate decay from validation
parser.add_argument('--test_interval', type=int, default=10, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int, default=100, help='Maximum number of epochs');
# ^^^ use --max_epoch=1 for local testing
parser.add_argument('--trainfunc', type=str, default="angleproto",    help='Loss function');
parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001,      help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');

## Loss functions
parser.add_argument("--hard_prob", type=float, default=0.5, help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank", type=int, default=10,    help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin', type=float,  default=0.3,     help='Loss margin, only for some loss functions');
parser.add_argument('--scale', type=float,   default=30,    help='Loss scale, only for some loss functions');
parser.add_argument('--nSpeakers', type=int, default=5994,  help='Number of speakers in the softmax layer for softmax-based losses, utterances per speaker per iteration for other losses');

## Load and save
parser.add_argument('--initial_model',  type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',      type=str, default="/tmp/data/exp1", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list', type=str, help='Train list');
parser.add_argument('--test_list',  type=str, help='Evaluation list');
parser.add_argument('--train_path', type=str, default="voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',  type=str, default="voxceleb1", help='Absolute path to the test set');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

## Model definition
parser.add_argument('--model', type=str,        default="ResNetSE34L",     help='Name of model definition');
parser.add_argument('--encoder_type', type=str, default="SAP",  help='Type of encoder');
parser.add_argument('--nOut', type=int,         default=512,    help='Embedding size in the last FC layer');

args = parser.parse_args();

print(args)

# set random seeds
# @TODO any reason to use BOTH 'random' and 'numpy.random'?
if args.set_seed:
    print("Using fixed random seed")
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

train_list, test_list, train_path, test_path = [None, None, None, None]

## Fetch data from GCS if enabled
if args.data_bucket is not None and not args.skip_data_fetch:
    print("Installing dataset from GCS")
    # @TODO mimic the --install-local-dataset function in
    #       data/utils.py, using the newer functions that it invokes
    #       in common/src/data_fetch.py

    # download, extract, transcode (compressed AAC->WAV) dataset
    download_gcs_dataset(args)
    extract_gcs_dataset(args)
    transcode_gcs_dataset(args)
    # set new lists and data paths
    train_list, test_list, train_path, test_path \
        = set_loc_paths_from_gcs_dataset(args)
elif args.data_bucket is not None and args.skip_data_fetch:
    print("Skipping GCS data fetch")
    # dataset from GCS already available; set new lists and data paths
    train_list, test_list, train_path, test_path \
        = set_loc_paths_from_gcs_dataset(args)
else:
    print("Using local, permanent dataset")
    # pass through to use permanent local dataset
    train_list = args.train_list
    test_list = args.test_list
    train_path = args.train_path
    test_path = args.test_path

# init directories
# temporary / internal output directories
tmp_output_dirs = [args.save_tmp_model_to, args.save_tmp_results_to,
        args.save_tmp_feats_to]
# directories of parmanent / component output artifacts
output_dirs = [os.path.dirname(args.save_model_to)]

for d in (tmp_output_dirs + output_dirs):
    if not(os.path.exists(d)):
        os.makedirs(d)

# set torch device to cuda or cpu
cuda_avail = torch.cuda.is_available()
print(f"Cuda available: {cuda_avail}")
use_cuda = cuda_avail and not args.no_cuda
print(f"Using cuda: {use_cuda}")
device = torch.device("cuda" if use_cuda else "cpu")

it          = 1;
prevloss    = float("inf");
sumloss     = 0;

# Load model weights

## Assertion
gsize_dict  = {'proto':args.nSpeakers, 'triplet':2, 'contrastive':2, 'softmax':1, 'amsoftmax':1, 'aamsoftmax':1, 'ge2e':args.nSpeakers, 'angleproto':args.nSpeakers}

assert args.trainfunc in gsize_dict
assert gsize_dict[args.trainfunc] <= 100

## Initialise data loader
trainLoader = DatasetLoader(train_list,
        gSize=gsize_dict[args.trainfunc], new_train_path=train_path,
        **vars(args));

torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512,
        win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

start_time = time.time()

# grab names of test and train from paths
test_name = args.test_path.replace(".tar.gz", "")
train_name = args.train_path.replace(".tar.gz", "")

extracted_feats_dataset_name = f"{train_name}_feats_{args.run_id}"

## print some spectrograms
#    mel_filter_bank = torchfb(batch.to(device))+1e-6
#    plt.figure()
#    #plt.imsave(f"/mnt/c/wsl-shared/voxsrc-2020/spec-{count}.png", mel_filter_bank.log()[0,:,:].numpy(), cmap='gray')
#    print(f"Current labels: {batch_labels}")
#    plt.imsave(f"/mnt/c/wsl-shared/voxsrc-2020/spec-1.png", mel_filter_bank.log()[0,:,:].numpy(), cmap='gray')
#    plt.figure()
#    plt.plot(batch[0,:].numpy())
#    plt.savefig(f"/mnt/c/wsl-shared/voxsrc-2020/vis-1.png")
#    count += 1
#    break

# @TODO Is there a package that does this without the inline update that tqdm
#       does? (which doesn't play nice with stackdrive kubeflow logs)
status_update_percentage = 10
status_update_interval = math.floor(len(trainLoader) * (status_update_percentage / 100))
status_batches_processed = 0

# @TODO use batches of utterance feats instead of single
for batch_utterance_feats, batch_utterance_paths in trainLoader:
    for utterance_index in range(0, len(batch_utterance_paths)):
        utterance_feats = batch_utterance_feats[utterance_index]
        utterance_path = batch_utterance_paths[utterance_index]

        mel_filter_bank = torchfb(utterance_feats.to(device))+1e-6
        log_mel_filter_bank = mel_filter_bank.cpu().log()

        # remove the ".wav" and add a _feats_[run id] to the name of the dataset
        spectrogram_file_name = utterance_path\
                .replace(".wav", "")\
                .replace(train_name, extracted_feats_dataset_name)
        # ensure path to file exists
        Path(os.path.dirname(spectrogram_file_name)).mkdir(parents=True,
                exist_ok=True)
        # save the spectrogram data to file
        numpy.save(spectrogram_file_name, log_mel_filter_bank.numpy())

    # status update
    if status_batches_processed % status_update_interval == 0:
        print(f"Processed {status_batches_processed}/{len(trainLoader)} "
              f"batches in {time.time() - start_time}")
        status_time = time.time()

    status_batches_processed += 1

print(f"Finished in {time.time() - start_time} seconds")
print(f"Extracted features saved to {extracted_feats_dataset_name}")
