#!/usr/bin/python3
from shutil import copyfile
import os
from tqdm import tqdm
import argparse
import subprocess
import sys

# Usage example:
#   python utils.py --action copy_train

parser = argparse.ArgumentParser(description='Voxceleb Dataset Utils')

# actions and their params
parser.add_argument('--copy-train', action='store_true')
parser.add_argument('--copy-test', action='store_true')

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

else:
    print(f"Invalid 'action' param: {args.action}")
