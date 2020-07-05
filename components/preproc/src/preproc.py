#!/usr/bin/env python3
import time
import argparse
from data_fetch import download_gcs_dataset, extract_gcs_dataset, \
                     transcode_gcs_dataset, set_loc_paths_from_gcs_dataset

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Data preprocessor')

parser.add_argument('--data-bucket', type=str)
parser.add_argument('--test_list', type=str)
parser.add_argument('--train_list', type=str)
parser.add_argument('--test_path', type=str)
parser.add_argument('--train_path', type=str)

parser.add_argument('--save-tmp-data-to', type=str, default="./tmp/data/")
parser.add_argument('--skip-data-fetch', action='store_true')
parser.add_argument('--save-tmp-model-to', type=str, default="./tmp/model/");
parser.add_argument('--save-tmp-results-to', type=str, default="./tmp/results/");
parser.add_argument('--save-tmp-feats-to', type=str, default="./tmp/feats/");

# permanent/component outputs
parser.add_argument('--save-model-to', type=str, default="./out/model.txt")
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
#upload_blob(args.data_bucket, )
