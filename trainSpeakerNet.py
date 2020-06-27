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
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

parser = argparse.ArgumentParser(description = "SpeakerNet");

## New args to support running on kubernetes/kubeflow
parser.add_argument('--model-save-path', type=str, default="model/");
# set this in order to fetch lists and data from GCS before reading
# them from local filesystem
parser.add_argument('--data-bucket', type=str)
parser.add_argument('--save-data-to', type=str, default="./tmp/")

## Data loader
parser.add_argument('--max_frames', type=int, default=200,  help='Input length to the network');
#parser.add_argument('--batch_size', type=int, default=30,  help='Batch size');
parser.add_argument('--batch_size', type=int, default=200,  help='Batch size');
parser.add_argument('--max_seg_per_spk', type=int, default=100, help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5, help='Number of loader threads');

## Training details
parser.add_argument('--test_interval', type=int, default=10, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int, default=1, help='Maximum number of epochs');
parser.add_argument('--trainfunc', type=str, default="amsoftmax",    help='Loss function');
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

times = []
## Fetch data from GCS if enabled
if args.data_bucket is not None:
    data_path = os.path.join(args.save_data_to,"data/")
    model_path = os.path.join(args.save_data_to,"model/")
    # make dir for data
    if not(os.path.exists(args.save_data_to)):
        os.makedirs(data_path)
        os.makedirs(model_path)

    # compose blob names
    list_blobs = [args.train_list, args.test_list]
    data_blobs = [args.train_path, args.test_path]
    blobs = list_blobs + data_blobs

    times.append({'start': time.time()})
    print("Downloading dataset blobs")
    # download each blob
    for blob in blobs:
        NUM_CORES = 8 # hard-coded to prod/cluster machine type
        src = f"gs://{args.data_bucket}/{blob}"
        dst = os.path.join(data_path, blob)
        subprocess.call(f"gsutil \
                            -o 'GSUtil:parallel_thread_count=1' \
                            -o 'GSUtil:sliced_object_download_max_components={NUM_CORES}' \
                            cp {src} {dst}", shell=True)
    times[-1]['stop'] = time.time()
    times[-1]['elapsed'] = times[-1]['stop'] - times[-1]['start']

    times.append({'start': time.time()})
    print(f"Uncompressing train/test data blobs")
    # uncompress data blobs
    for blob in tqdm(data_blobs):
        dst = os.path.join(data_path, blob)
        with open(os.devnull, 'w') as FNULL:
            subprocess.call(f"tar -C {data_path} -zxvf {dst}",
                    shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    times[-1]['stop'] = time.time()
    times[-1]['elapsed'] = times[-1]['stop'] - times[-1]['start']

    times.append({'start': time.time()})
    # convert train data from AAC (.m4a) to WAV (.wav)
    # @note Didn't compress the test data--wasn't originally provided
    #       in compressed form and wasn't sure if compressing w/ lossy
    #       AAC would degrade audio relative to blind test set
    # @TODO try lossy-compressing voxceleb1 test data w/ AAC
    for blob in [args.train_path]:
        # get full path to blob's uncompressed data dir
        blob_dir_name = args.train_path.split('.tar.gz')[0]
        blob_dir_path = os.path.join(data_path, blob_dir_name)
        # get list of all nested files
        files = glob.glob(f"{blob_dir_path}/*/*/*.m4a")

        # @note Achieved best transcoding/audio-decompression results
        #       using parallel, rather than multiple python processes
        USE_PARALLEL = True
        if USE_PARALLEL:
            # @TODO confirm that this is IO-bound and no additional
            #       CPU-usage is possible
            print("Writing wav files names to file")
            transcode_list_path = os.path.join(data_path, "transcode-list.txt")
            with open(transcode_list_path, "w") as outfile:
                outfile.write("\n".join(files))
            print("Constructing command")
            cmd = "cat %s | parallel ffmpeg -y -i {} -ac 1 -vn \
                    -acodec pcm_s16le -ar 16000 {.}.wav >/dev/null \
                    2>/dev/null" % (transcode_list_path)
            print("Uncompressing audio AAC->WAV in parallel...")
            subprocess.call(cmd, shell=True)
        else:
            print(f"Compiling '{blob_dir_name}' convert cmd list")
            cmd_list = [f"ffmpeg -y -i {filename} -ac 1 -vn -acodec \
                          pcm_s16le -ar 16000 {filename.replace('.m4a', '.wav')} \
                          >/dev/null 2>/dev/null"
                        for filename in tqdm(files)
                       ]

            print(f"Converting '{blob_dir_name}' files from AAC to WAV")
            pool = Pool(4) # num concurrent threads
            for i, returncode in tqdm(enumerate(pool.imap(partial(subprocess.call, shell=True), cmd_list))):
                if returncode != 0:
                    print("%d command failed: %d" % (i, returncode))

    times[-1]['stop'] = time.time()
    times[-1]['elapsed'] = times[-1]['stop'] - times[-1]['start']

print(f"Downloaded, extracted, converted in: {times}")

## Initialise directories
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"
feat_save_path      = ""

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

## Load models
s = SpeakerNet(**vars(args));

it          = 1;
prevloss    = float("inf");
sumloss     = 0;

## Load model weights
modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(0,it-1):
    if ii % args.test_interval == 0:
        clr = s.updateLearningRate(args.lr_decay) 

## Evaluation code
if args.eval == True:
        
    sc, lab = s.evaluateFromListSave(args.test_list, print_interval=100, feat_dir=feat_save_path, test_path=args.test_path)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    print('EER %2.4f'%result[1])

    quit();

## Write args to scorefile
scorefile = open(result_save_path+"/scores.txt", "a+");

for items in vars(args):
    print(items, vars(args)[items]);
    scorefile.write('%s %s\n'%(items, vars(args)[items]));
scorefile.flush()

## Assertion
gsize_dict  = {'proto':args.nSpeakers, 'triplet':2, 'contrastive':2, 'softmax':1, 'amsoftmax':1, 'aamsoftmax':1, 'ge2e':args.nSpeakers, 'angleproto':args.nSpeakers}

assert args.trainfunc in gsize_dict
assert gsize_dict[args.trainfunc] <= 100

## Initialise data loader
trainLoader = DatasetLoader(args.train_list, gSize=gsize_dict[args.trainfunc], **vars(args));

clr = s.updateLearningRate(1)

# touch the output file/dir
print("Creating parent dir for path={args.model_save_path}")
Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)

while(1):   
    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %f..."%(args.model,max(clr)));

    ## Train network
    loss, traineer = s.train_network(loader=trainLoader);

    ## Validate and save
    if it % args.test_interval == 0:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");

        sc, lab = s.evaluateFromListSave(args.test_list, print_interval=100, feat_dir=feat_save_path, test_path=args.test_path)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/T1 %2.2f, TLOSS %f, VEER %2.4f"%( max(clr), traineer, loss, result[1]));
        scorefile.write("IT %d, LR %f, TEER/T1 %2.2f, TLOSS %f, VEER %2.4f\n"%(it, max(clr), traineer, loss, result[1]));

        scorefile.flush()

        clr = s.updateLearningRate(args.lr_decay) 

        s.saveParameters(model_save_path+"/model%09d.model"%it);
        
        ## touch the output file/dir
        #Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        #with open(args.model_save_path, 'w') as eerfile:
        #    eerfile.write(f"model iter: {it}")
        #    eerfile.write('%.4f'%result[1])
            
        eerfile = open(model_save_path+"/model%09d.eer"%it, 'w')
        eerfile.write('%.4f'%result[1])
        eerfile.close()
        ret = '%.4f'%result[1]

    else:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER %2.2f, TLOSS %f"%( max(clr), traineer, loss));
        scorestuff = "IT %d, LR %f, TEER %2.2f, TLOSS %f\n"%(it, max(clr), traineer, loss)
        scorefile.write(scorestuff);
        # write contents
        with open(args.model_save_path, 'w') as model_save_file:
            model_save_file.write(f"[model] ret={scorestuff}\n")

        scorefile.flush()

    if it >= args.max_epoch:
        quit();

    it+=1;
    print("");

scorefile.close();





