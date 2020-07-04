#!/usr/bin/env python3
import time
import argparse
from data_fetch import download_gcs_dataset, extract_gcs_dataset, \
                     transcode_gcs_dataset, set_loc_paths_from_gcs_dataset

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Data preprocessor')

parser.add-argument('--data-bucket', type=str)
parser.add-argument('--test-list', type=str)
parser.add-argument('--train-list', type=str)
parser.add-argument('--test-path', type=str)
parser.add-argument('--train-path', type=str)
args = parser.parse_args()

# download,extract,transcode dataset blobs from GCS
download_gcs_dataset(args)
extract_gcs_dataset(args)
transcode_gcs_dataset(args)

# set new lists and data paths
train_list, test_list, train_path, test_path \
    = set_loc_paths_from_gcs_dataset(args)

# perform trivial operation on audio data
time.sleep(1)
print("TODO")

# recompress/transcode
print("TODO")

# archive into tar
print("TODO")

# upload dataset blobs to GCS
