#!/usr/bin/python3

# @TODO clean up naming to distinuish this dev environment dataset utilities
#       file from the component source data utilities for the actual ML pipeline

import sys

# Add common src dir to python import path (varies between runs on and
# off the training cluster)
sys.path.insert(1, "../common/src/")

from shutil import copyfile
import os
from tqdm import tqdm
import argparse
import subprocess
from pathlib import Path
import time

from data_utils import download_gcs_blob_in_parallel, extract_tar, \
                     convert_aac_to_wav

# Usage example:
#   python utils.py --action copy_train

parser = argparse.ArgumentParser(description='Voxceleb Dataset Utils')

# copy a subset of the training data
parser.add_argument('--copy-train', action='store_true')

# copy a subset of the testing data
parser.add_argument('--copy-test', action='store_true')

# download, extract, and transcode data from GCS for outside of the
# cluster
parser.add_argument('--install-local-dataset', action='store_true')
parser.add_argument('--src-bucket', required=('--install-local-dataset' in sys.argv))
parser.add_argument('--src-dataset', required=('--install-local-dataset' in sys.argv))
parser.add_argument('--dst-data-path', required=('--install-local-dataset' in sys.argv))
parser.add_argument('--dst-list-path', required=('--install-local-dataset' in sys.argv))
parser.add_argument('--dst-tmp-path', default="./")

# generate a test_utterance_list from a test_path
parser.add_argument('--generate-test-utterance-list', action='store_true')
parser.add_argument('--test-path', required=('--generate-test-utterance-list' in sys.argv))

# compress a dataset
parser.add_argument('--compress', action='store_true', help="requires --dir=[path to directory to compress]")
parser.add_argument('--src-dir', required=('--compress' in sys.argv))

# common-required args
needs_src_list = '--copy-train' in sys.argv or '--copy-test' in sys.argv
parser.add_argument('--src-list', required=needs_src_list)

needs_dst_dir = '--copy-train' in sys.argv or '--copy-test' \
        in sys.argv or '--compress' in sys.argv
parser.add_argument('--dst-dir', required=needs_dst_dir)

args = parser.parse_args()

# local path to full voxceleb2
TRAIN_SRC_DIR = "/home/voxceleb/voxceleb2/"
# local path to full voxceleb1
TEST_SRC_DIR = "/home/voxceleb/voxceleb1/"

# training data
# example usage:
# ------------------------------
# python utils.py --copy-train --src-list lists/vox2_no_cuda.txt --dst-dir ./datasets/vox2_no_cuda
# ------------------------------
if args.copy_train:
    print(f"Copying a train dataset")
    # @note voxceleb2 (train) has .m4a (AAC), while voxceleb1 (test) has .wav
    EXT = "m4a"

    # read train file list
    with open(args.src_list) as f:
        lines = f.readlines()
        src_paths_no_ext = [line.split(' ')[1].split('\n')[0].split('wav')[0] for line in lines]

        for src_path_no_ext in tqdm(src_paths_no_ext):
            src_path = src_path_no_ext + EXT
            src = os.path.join(TRAIN_SRC_DIR, src_path)
            dst = os.path.join(args.dst_dir, src_path)

            # create destination dir if doesn't exist and copy file
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            copyfile(src, dst)
        print(f"Copied {len(src_paths_no_ext)} training .{EXT} files.")

# @NOTE Not using aac/m4a compressed files for test set for fear of
#       degrading audio quality. Not sure if the lossy compression
#       scheme would further reduce audio quality (assuming it's
#       already been compressed/uncompressed w/ lossy encoding before),
#       or that it was even compressed to begin with. Could result in
#       a difficult-to-debug difference between validation and
#       blind-test accuracy down the road.

# test data
# example usage:
# ------------------------------
# python utils.py --copy-test --src-list lists/vox1_no_cuda.txt --dst-dir ./datasets/vox1_no_cuda
# ------------------------------
elif args.copy_test:
    print(f"Copying a test dataset")
    EXT = "wav"

    with open(args.src_list) as f:
        lines = f.readlines()
        # add first utterance in test pair
        src_paths_no_ext = [line.split(' ')[1].split('wav')[0] for line in lines]
        # add second utterance in test pair
        src_paths_no_ext.extend(
          [line.split(' ')[2].split('\n')[0].split('wav')[0] for line in lines])

        # copy all data from source
        for src_path_no_ext in tqdm(src_paths_no_ext):
            src_path = src_path_no_ext + EXT
            src = os.path.join(TEST_SRC_DIR, src_path)
            dst = os.path.join(args.dst_dir, src_path)

            # check if already copied it (many duplicates across test pairs)
            if not os.path.isfile(dst):
                # create destination dir if doesn't exist and copy file
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                copyfile(src, dst)
        print(f"Copied some subset of {len(src_paths_no_ext)} non-unique test files.")

# example usage:
# ------------------------------
# python utils.py --compress --src-dir datasets/vox1_no_cuda --dst-dir ./tars
# ------------------------------
elif args.compress:
    raise "ERROR: WIP, actually is broken--doesn't work if in nest dir\
            . Need something like tar -C ./datasets/ -zcvf tars/vox1_no_cuda.tar.gz vox1_no_cuda"
    dataset_name = os.path.basename(args.src_dir)
    if dataset_name == "":
        raise "ERROR: No dataset name, did you put a trailing slash?"
    # make tar dir if not present
    if not(os.path.exists(args.dst_dir)):
        os.makedirs(args.dst_dir)
    # tar it up
    dst_file = os.path.join(args.dst_dir, dataset_name + ".tar.gz")
    print(f"Destination file: {dst_file}")
    subprocess.call(f"tar -zcvf {dst_file} {args.src_dir}", shell=True)

# @example
#
#     python utils.py --install-local-dataset --src-bucket \
#        voxsrc-2020-voxceleb-v4 --src-dataset no_cuda --dst-data-path \
#        ./datasets --dst-list-path ./lists --dst-tmp-path ./tmp
#
# @note Current datasets: no_cuda, full
elif args.install_local_dataset:
    print(f"Installing local dataset")

    # create directories if not present
    for path in [args.dst_data_path, args.dst_list_path, \
            args.dst_tmp_path]:
        if not(os.path.exists(path)):
            print(f"Creating directory: {path}")
            os.makedirs(path)

    # download all list blobs to the list folder
    for blob in [f"vox2_{args.src_dataset}.txt",
                 f"vox1_{args.src_dataset}.txt"]:
        download_gcs_blob_in_parallel(args.src_bucket, blob,
                args.dst_list_path)

    # download all archived data blobs to the tmp dir and then
    # unarchive them into the data dir
    for blob in [f"vox2_{args.src_dataset}.tar.gz",
                 f"vox1_{args.src_dataset}.tar.gz"]:
        download_gcs_blob_in_parallel(args.src_bucket, blob,
                args.dst_tmp_path)
        extract_tar(os.path.join(args.dst_tmp_path, blob),
                args.dst_data_path)

    # convert compressed training audio data from AAC(.m4a) to WAV(.wav)
    aac_train_data_path = os.path.join(args.dst_data_path,
            f"vox2_{args.src_dataset}/")
    convert_aac_to_wav(aac_train_data_path, args.dst_tmp_path)

    print("************  NOTICE ME  ************\n"
           "-> You must now symlink these paths with (for example):\n"
           "   ln -s ./datasets/vox1_no_cuda ../components/train/tmp/data/vox1_no_cuda\n"
           "   ln -s ./datasets/vox2_no_cuda ../components/train/tmp/data/vox2_no_cuda\n"
           "   ln -s ./datasets/vox1_no_cuda.txt ../components/train/tmp/data/vox1_no_cuda.txt\n"
           "   ln -s ./datasets/vox2_no_cuda.txt ../components/train/tmp/data/vox2_no_cuda.txt\n")

# Example usage
#   python utils.py --generate-test-utterance-list --test-path \
#        datasets/vox1_no_cuda/
elif args.generate_test_utterance_list:
    print(f"Generating test utterance list from {args.test_path}")
    start_time = time.time()

    out_file_name = 'datasets/new_test_utterance_list.txt'
    with open(out_file_name, 'w') as f:
        for path in Path(args.test_path).rglob('*.wav'):
            path_str = str(path).replace(args.test_path, '')
            speaker_id = path_str.split('/')[0]
            f.write(f"{speaker_id} {path_str}\n")
    print(f"Wrote utterance list (in train data style) to {out_file_name} in {time.time() - start_time} (s)")

else:
    print(f"Invalid 'action' param: {args.action}")
