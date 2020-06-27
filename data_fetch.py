from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import subprocess
import time
import glob
import os

# @brief Collection of functions for downloading, extracting, and
#        transcoding/uncompressing a dataset stored on GCS

def download_gcs_dataset(args):
    start = time.time()
    print("Downloading dataset blobs...")

    # make directories for tmp data
    if not(os.path.exists(args.save_tmp_data_to)):
        os.makedirs(args.save_tmp_data_to)
    if not(os.path.exists(args.save_tmp_model_to)):
        os.makedirs(args.save_tmp_model_to)

    # compose blob names
    list_blobs = [args.train_list, args.test_list]
    data_blobs = [args.train_path, args.test_path]
    blobs = list_blobs + data_blobs

    # download each blob
    for blob in blobs:
        NUM_CORES = 8 # hard-coded to prod/cluster machine type
        src = f"gs://{args.data_bucket}/{blob}"
        dst = os.path.join(args.save_tmp_data_to, blob)
        subprocess.call(f"gsutil \
                            -o 'GSUtil:parallel_thread_count=1' \
                            -o 'GSUtil:sliced_object_download_max_components={NUM_CORES}' \
                            cp {src} {dst}", shell=True)
    print(f"...Finished in {time.time() - start} (s)")

def extract_gcs_dataset(args):
    start = time.time()
    print(f"Uncompressing train/test data blobs...")

    data_blobs = [args.train_path, args.test_path]

    # uncompress data blobs
    for blob in tqdm(data_blobs):
        dst = os.path.join(args.save_tmp_data_to, blob)
        with open(os.devnull, 'w') as FNULL:
            subprocess.call(f"tar -C {args.save_tmp_data_to} -zxvf {dst}",
                    shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    print(f"...Finished in {time.time() - start} (s)")

def transcode_gcs_dataset(args):
    start = time.time()
    print(f"Transcoding training data from AAC->WAV...")
    # convert train data from AAC (.m4a) to WAV (.wav)
    # @note Didn't compress the test data--wasn't originally provided
    #       in compressed form and wasn't sure if compressing w/ lossy
    #       AAC would degrade audio relative to blind test set
    # @TODO try lossy-compressing voxceleb1 test data w/ AAC
    for blob in [args.train_path]:
        # get full path to blob's uncompressed data dir
        blob_dir_name = args.train_path.split('.tar.gz')[0]
        blob_dir_path = os.path.join(args.save_tmp_data_to, blob_dir_name)
        # get list of all nested files
        files = glob.glob(f"{blob_dir_path}/*/*/*.m4a")

        # @note Achieved best transcoding/audio-decompression results
        #       using parallel, rather than multiple python processes
        USE_PARALLEL = True
        if USE_PARALLEL:
            # @TODO confirm that this is IO-bound and no additional
            #       CPU-usage is possible
            print("Writing wav file names to file")
            transcode_list_path = os.path.join(args.save_tmp_data_to,
                    "transcode-list.txt")
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

    print(f"...Finished in {time.time() - start} (s)")

def set_loc_paths_from_gcs_dataset(args):
    # set new lists and data paths
    train_list = os.path.join(args.save_tmp_data_to, args.train_list)
    test_list = os.path.join(args.save_tmp_data_to, args.train_list)
    # @note remove the .tar.gz to reference extracted directories
    train_path = os.path.join(args.save_tmp_data_to,
            args.train_path.split(".tar.gz")[0])
    test_path = os.path.join(args.save_tmp_data_to,
            args.test_path.split(".tar.gz")[0])
    return train_list, test_list, train_path, test_path
