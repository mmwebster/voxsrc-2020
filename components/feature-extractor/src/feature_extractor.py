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
from FeatureExtractor import FeatureExtractor
import subprocess
import time
from pathlib import Path
from utils.data_utils import download_gcs_dataset, extract_gcs_dataset, \
                     transcode_gcs_dataset, get_loc_paths_from_gcs_dataset,\
                     download_blob, upload_blob, compress_to_tar
import yaml
import pwd
import google
import wandb

# @TODO Use a logger instead of print statements

# @brief Generate a unique run ID when not run in kubeflow (kubeflow passes
#        its own default run ID) in order to store training artifacts
#        for resume after preemption
# @note To manually resume a run outside of kubeflow, pass the run ID
#       printed in the run with the "--run-id" flag
def gen_run_id():
    user_id = pwd.getpwuid( os.getuid() )[ 0 ]
    wandb_id = wandb.util.generate_id()
    return f"{user_id}-{wandb_id}"

# @brief write outputs to provided output paths
def write_output_artifact(path, value):
    # @TODO this is currently going around kubeflow's built in mechanisms. Wasn't
    #       sure if a component outputPath could be read and delivered to downstream
    #       components as a string, int, etc, rather than as a file path. Figure out
    #       the right way to do this...
    # open and write
    with open(path, 'w') as f:
        f.write(value)

parser = argparse.ArgumentParser(description = "Feature Extractor");

# @note "tmp" denotes that this component output data will not be captured by
#       the kubeflow pipeline or made available to downstream components
parser.add_argument('--data-bucket', type=str,
        help="Name of GCS bucket to use for this run. E.g. voxsrc-2020-voxceleb-v4")
parser.add_argument('--save-tmp-data-to', type=str, default="./tmp/data/",
        help="Relative path to output data, temporarily stored on cluster, "
             "permanenly written on local. E.g. ./tmp/data/")
parser.add_argument('--no-cuda', action='store_true',
        help="Flag to disable cuda for this run");
parser.add_argument('--set-seed', action='store_true',
        help="Flag to set a random seed for random, numpy.random, and torch");
parser.add_argument('--no-upload', action='store_true',
        help="Flag to disable upload of output artifacts to GCS "
             "(speeds up local runs)")
parser.add_argument('--output-path-test-feats-tar-path', type=str,
        default="./tmp/outputs/test_feats_tar_path",
        help="Path to file that stores the GCS output artifact path of test "
             "features for cluster runs. There is no need to modify this.")
parser.add_argument('--output-path-train-feats-tar-path', type=str,
        default="./tmp/outputs/train_feats_tar_path",
        help="Path to file that stores the GCS output artifact path of train "
             "features for cluster runs. There is no need to modify this.")
parser.add_argument('--num-threads', type=int, default=10,
        help="The number of worker threads that the feature extractor uses to "
             "boost IO-bound performance. E.g. 10")
parser.add_argument('--reuse-run-with-id', type=str, default="",
        help="Execute the component in pass-through mode. Output all "
             "expected outputs, but using GCS artifacts from a previous run, "
             "with the provided ID. E.g. 1a373cca-7db8-4dd0-bec8-79a83041458b")
parser.add_argument('--run-id', type=str, default=f"{gen_run_id()}",
        help="Auto-generated unique run ID. There is no need to modify this.")
parser.add_argument('--train_list', type=str,
        help='GCS path to train list. E.g. vox2_no_cuda.txt');
parser.add_argument('--test_utterances_list',  type=str,
        help="GCS path to test utterances list. Note that this file does not "
             "containing testing pairs like 'test_list' in the train component "
             "does. This only lists the utterances like in the 'train_list'. "
             "E.g. vox1_no_cuda_utterances.txt");
parser.add_argument('--train_path', type=str, default="voxceleb2",
        help="GCS path to compressed train dataset with .tar.gz file extension. "
             "E.g. vox2_no_cuda.tar.gz");
parser.add_argument('--test_path',  type=str, default="voxceleb1",
        help="GCS path to compressed test dataset with .tar.gz file extension. "
             "E.g. vox1_no_cuda.tar.gz");

args = parser.parse_args();

print(args)

start_time = time.time()

# ensure all output directories exist
output_dirs = [args.save_tmp_data_to,
              os.path.dirname(args.output_path_train_feats_tar_path),
              os.path.dirname(args.output_path_test_feats_tar_path)]
for dir in output_dirs:
    Path(dir).mkdir(parents=True, exist_ok=True)

if not args.reuse_run_with_id:
    # set random seeds
    # @TODO any reason to use BOTH 'random' and 'numpy.random'?
    if args.set_seed:
        print("Using fixed random seed")
        random.seed(0)
        numpy.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

    train_list, test_utterances_list, train_path, test_path = [None, None, None, None]

    print("Installing dataset from GCS")
    # @TODO mimic the --install-local-dataset function in
    #       data/utils.py, using the newer functions that it invokes
    #       in common/src/utils/data_utils.py
    # @TODO change extract and transcode to take explicit params

    # download, extract, transcode (compressed AAC->WAV) dataset
    blobs = [args.train_list, args.test_utterances_list,
            args.train_path, args.test_path]
    download_gcs_dataset(args.data_bucket, args.save_tmp_data_to, blobs)
    extract_gcs_dataset(args, use_pigz=True)
    transcode_gcs_dataset(args)
    # set new lists and data paths
    train_list, test_utterances_list, train_path, test_path \
        = get_loc_paths_from_gcs_dataset(args.save_tmp_data_to, blobs)

    # set torch device to cuda or cpu
    cuda_avail = torch.cuda.is_available()
    print(f"Cuda available: {cuda_avail}")
    use_cuda = cuda_avail and not args.no_cuda
    print(f"Using cuda: {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512,
            win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

    def feature_extractor_fn(utterance_wav):
        mel_filter_bank = torchfb(utterance_wav.to(device))+1e-6
        log_mel_filter_bank = mel_filter_bank.cpu().log()
        return log_mel_filter_bank.numpy().astype('float16')

    # compose list and data paths
    datasets = []
    # grab names/paths of test
    test_name = args.test_path.replace(".tar.gz", "")
    extracted_test_feats_dataset_name = f"{test_name}_feats_{args.run_id}"
    dst_test_feats_path = os.path.join(args.save_tmp_data_to, extracted_test_feats_dataset_name)
    datasets.append({"list_path": test_utterances_list, "data_path": test_path,
        "dst_feats_path": dst_test_feats_path,
        "extracted_feats_dataset_name": extracted_test_feats_dataset_name,
        "artifact_output_path": args.output_path_test_feats_tar_path})
    # grab names/paths of train
    train_name = args.train_path.replace(".tar.gz", "")
    extracted_train_feats_dataset_name = f"{train_name}_feats_{args.run_id}"
    dst_train_feats_path = os.path.join(args.save_tmp_data_to, extracted_train_feats_dataset_name)
    datasets.append({"list_path": train_list, "data_path": train_path,
        "dst_feats_path": dst_train_feats_path,
        "extracted_feats_dataset_name": extracted_train_feats_dataset_name,
        "artifact_output_path": args.output_path_train_feats_tar_path})

    for dataset in datasets:
        # init the train dataset feature extractor and run it
        with FeatureExtractor(dataset["list_path"], dataset["data_path"],
                dataset["dst_feats_path"], feature_extractor_fn,
                num_threads = args.num_threads) as feature_extractor:
            print(f"Starting feature extraction for "
                  f"{dataset['data_path']}:{dataset['list_path']}")
            feature_extractor.run()

        # write arg parse params to metadata.txt
        metadata_file_path = os.path.join(args.save_tmp_data_to,
                dataset["extracted_feats_dataset_name"], 'metadata.txt')
        # ensure exists
        Path(os.path.dirname(metadata_file_path)).mkdir(parents=True, exist_ok=True)

        with open(metadata_file_path, "w") as f:
            # add arg parse params
            for items in vars(args):
                f.write(f"{items}: {vars(args)[items]}\n")
            # add git state
            try:
                # commit hash
                git_hash_dirty = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
                git_hash_clean = git_hash_dirty.decode('utf8').replace('\n','')
                # clean/dirty status of local git
                git_status_dirty = subprocess.check_output(['git', 'diff', '--stat'])
                git_status_clean = git_status_dirty.decode('utf8')
                git_status = 'clean' if git_status_clean == "" else 'dirty'
                # write them
                f.write(f"Git commit: {git_hash_clean}\n")
                f.write(f"Git status: {git_status}\n")
            except subprocess.CalledProcessError:
                f.write(f"Git commit: [N/A... on cluster]\n")
                f.write(f"Git status: [N/A... on cluster]\n")

        # tar up the result
        dst_feats_path_without_trailing_slash = os.path.join(dataset["dst_feats_path"], '')[:-1]
        dst_tar_file_path = dst_feats_path_without_trailing_slash + '.tar.gz'
        compress_to_tar(dataset["dst_feats_path"], dst_tar_file_path, use_pigz=True)

        # upload the tar to GCS in data_bucket at top level
        dst_tar_blob_path = dataset["extracted_feats_dataset_name"] + '.tar.gz'
        if not args.no_upload:
            upload_blob(args.data_bucket, dst_tar_blob_path, dst_tar_file_path)

        # write GCS paths to file for cluster kubeflow component output artifact
        write_output_artifact(path=dataset["artifact_output_path"], value=dst_tar_blob_path)

        print(f"Extracted features saved to {dataset['dst_feats_path']}")
        print(f"Tar file saved to {dst_tar_file_path}")
else:
    # write pass through artifacts to files for component outputs
    # test
    test_name = args.test_path.replace(".tar.gz", "")
    dst_test_tar_blob_path = f"{test_name}_feats_{args.reuse_run_with_id}.tar.gz"
    # train
    train_name = args.train_path.replace(".tar.gz", "")
    dst_train_tar_blob_path = f"{train_name}_feats_{args.reuse_run_with_id}.tar.gz"
    # write
    write_output_artifact(path=args.output_path_test_feats_tar_path,
            value=dst_test_tar_blob_path)
    write_output_artifact(path=args.output_path_train_feats_tar_path,
            value=dst_train_tar_blob_path)

print(f"Finished in {time.time() - start_time} (s)")
