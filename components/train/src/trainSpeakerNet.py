#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import numpy
import pdb
import torch
import glob
from tuneThreshold import tuneThresholdfromScore
from SpeakerNet import SpeakerNet
from DatasetLoader import DatasetLoader
import subprocess
import time
from pathlib import Path
from data_fetch import download_gcs_dataset, extract_gcs_dataset, \
                     transcode_gcs_dataset, set_loc_paths_from_gcs_dataset,\
                     download_blob, upload_blob
import yaml
import os
import pwd
import google
import wandb

# for local runs, use this for unique checkpoint dirs across team members
def get_username():
    return pwd.getpwuid( os.getuid() )[ 0 ]

parser = argparse.ArgumentParser(description = "SpeakerNet");

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

parser.add_argument('--checkpoint-bucket', type=str,
        default="voxsrc-2020-checkpoints-dev");
parser.add_argument('--checkpoint-path', type=str, default=f"{get_username()}/");

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

wandb.init(project="voxsrc-2020-v1", config=args)

train_list, test_list, train_path, test_path = [None, None, None, None]

## Fetch data from GCS if enabled
if args.data_bucket is not None and not args.skip_data_fetch:
    print("Performing GCS data fetch")
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

# set device cuda or cpu
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
print(f"Cuda available: {cuda_avail}")

## Load models
s = SpeakerNet(device, **vars(args));

it          = 1;
prevloss    = float("inf");
sumloss     = 0;

# Load model weights

# Check for training meta data from a previously preempted run
METADATA_NAME = 'metadata.yaml'
metadata_gcs_src_path = os.path.join(args.checkpoint_path, METADATA_NAME)
metadata_file_dst_path = os.path.join(args.save_tmp_model_to, METADATA_NAME)
default_metadata = {'is_done': False}
metadata = default_metadata

if args.reset_training:
    print("Starting at epoch 0, regardless of previous training progress")
else:
    # fetch metadata if available
    try:
        download_blob(args.checkpoint_bucket, metadata_gcs_src_path,
                metadata_file_dst_path)
        print("Downloaded previous training metadata")
        with open(metadata_file_dst_path, 'r') as f:
            try:
                metadata = yaml.safe_load(f)
                print(f"Loaded previous training metadata: {metadata}")
                # grab the latest model name (corresponding to the last epoch)
                latest_model_name = metadata['latest_model_name']
                # download the model
                model_gcs_src_path = os.path.join(args.checkpoint_path, latest_model_name)
                model_file_dst_path = os.path.join(args.save_tmp_model_to, latest_model_name)
                try:
                    download_blob(args.checkpoint_bucket, model_gcs_src_path,
                            model_file_dst_path)
                    print("**Downloaded a saved model**")
                    # load the saved model's params into the model class
                    s.loadParameters(model_file_dst_path);
                    print("Model %s loaded from previous state!"%model_file_dst_path);
                    it = int(os.path.splitext(os.path.basename(model_file_dst_path))[0][5:]) + 1
                except google.cloud.exceptions.NotFound:
                    print("**No saved model found**")
            except yaml.YAMLError as exc:
                print(exc)
                metadata = default_metadata
    except google.cloud.exceptions.NotFound:
        print("**No previous training metadata found**")

    # exit if previous training run finished
    if 'is_done' in metadata and metadata['is_done']:
        print("Terminating... training for this run has already completed")
        quit()

if(args.initial_model != ""):
    raise "Error: TODO"
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(0,it-1):
    if ii % args.test_interval == 0:
        clr = s.updateLearningRate(args.lr_decay) 

## Evaluation code
if args.eval == True:
    raise "Error: TODO"
    sc, lab = s.evaluateFromListSave(test_list, print_interval=100, feat_dir=args.save_tmp_feats_to, test_path=test_path)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    print('EER %2.4f'%result[1])

    quit();

## Write args to scorefile
scorefile = open(os.path.join(args.save_tmp_results_to,"scores.txt"), "a+");

for items in vars(args):
    print(items, vars(args)[items]);
    scorefile.write('%s %s\n'%(items, vars(args)[items]));
scorefile.flush()

## Assertion
gsize_dict  = {'proto':args.nSpeakers, 'triplet':2, 'contrastive':2, 'softmax':1, 'amsoftmax':1, 'aamsoftmax':1, 'ge2e':args.nSpeakers, 'angleproto':args.nSpeakers}

assert args.trainfunc in gsize_dict
assert gsize_dict[args.trainfunc] <= 100

## Initialise data loader
trainLoader = DatasetLoader(train_list,
        gSize=gsize_dict[args.trainfunc], new_train_path=train_path,
        **vars(args));

clr = s.updateLearningRate(1)

# touch the output file/dir
print(f"Creating parent dir for path={args.save_tmp_model_to}")
Path(args.save_tmp_model_to).parent.mkdir(parents=True, exist_ok=True)

while(1):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %f..."%(args.model,max(clr)));

    ## Train network
    loss, traineer = s.train_network(loader=trainLoader);

    wandb.log({'epoch': it, 'loss': loss, 'train_EER': traineer, 'lr': clr})

    ## Validate, save, update learning rate
    if it % args.test_interval == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");

        sc, lab = s.evaluateFromListSave(test_list, print_interval=100,
                feat_dir=args.save_tmp_feats_to, test_path=test_path)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/T1 %2.2f, TLOSS %f, VEER %2.4f"%( max(clr), traineer, loss, result[1]));
        scorefile.write("IT %d, LR %f, TEER/T1 %2.2f, TLOSS %f, VEER %2.4f\n"%(it, max(clr), traineer, loss, result[1]));

        scorefile.flush()

        clr = s.updateLearningRate(args.lr_decay) 


        ## touch the output file/dir
        #Path(args.save_tmp_model_to).parent.mkdir(parents=True, exist_ok=True)
        #with open(args.save_tmp_model_to, 'w') as eerfile:
        #    eerfile.write(f"model iter: {it}")
        #    eerfile.write('%.4f'%result[1])

        eerfile = open(args.save_tmp_model_to+"/model%09d.eer"%it, 'w')
        eerfile.write('%.4f'%result[1])
        eerfile.close()
        ret = '%.4f'%result[1]

    else:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER %2.2f, TLOSS %f"%( max(clr), traineer, loss));
        scorestuff = "IT %d, LR %f, TEER %2.2f, TLOSS %f\n"%(it, max(clr), traineer, loss)
        scorefile.write(scorestuff);
        # write contents
        with open(args.save_model_to, 'w') as model_save_file:
            model_save_file.write(f"[model] ret={scorestuff}\n")

        scorefile.flush()

    # save param dict for this epoch
    model_name = "model%09d.model"%it
    model_filename = os.path.join(args.save_tmp_model_to, model_name)
    s.saveParameters(model_filename);
    
    # save model for this epoch
    model_name = "model%09d.pt"%it
    model_filename = os.path.join(args.save_tmp_model_to, model_name)
    s.saveModel(model_filename);

    # update metadata
    metadata['latest_model_name'] = model_name
    metadata['num_epochs'] = it
    # dump metadata to yaml file
    with open(metadata_file_dst_path, 'w') as f:
        try:
            yaml.dump(metadata, f)
            print("Saved current training metadata")
        except yaml.YAMLError as exc:
            print(exc)
    # upload model to GCS
    model_gcs_dst_path = os.path.join(args.checkpoint_path, model_name)
    model_file_src_path = os.path.join(args.save_tmp_model_to, model_name)
    upload_blob(args.checkpoint_bucket, model_gcs_dst_path,
            model_file_src_path)
    # upload metadata to GCS
    metadata_gcs_dst_path = metadata_gcs_src_path
    metadata_file_src_path = metadata_file_dst_path
    upload_blob(args.checkpoint_bucket, metadata_gcs_dst_path,
            metadata_file_src_path)

    if it >= args.max_epoch:
        break

    it+=1;
    print("");

scorefile.close();


# save "done" to metadata so it restarts on retry
metadata['is_done'] = True

# dump metadata to yaml file
with open(metadata_file_dst_path, 'w') as f:
    try:
        yaml.dump(metadata, f)
        print("Saved current training metadata")
    except yaml.YAMLError as exc:
        print(exc)

# upload metadata to GCS
metadata_gcs_dst_path = metadata_gcs_src_path
metadata_file_src_path = metadata_file_dst_path
upload_blob(args.checkpoint_bucket, metadata_gcs_dst_path,
        metadata_file_src_path)
