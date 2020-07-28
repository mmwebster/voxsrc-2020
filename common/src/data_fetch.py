from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import subprocess
import time
import glob
import os
from google.cloud import storage
from google.auth import compute_engine

NUM_CORES_DEFAULT = 4

# @TODO Reorganize this as a "gcs_utils.py" and abstract away any
#       dataset-specific and non-gcs-specific stuff

def get_storage_client():
    # grab storage client, with credentials from compute engine unless
    # the google sdk credentials env var is set (like what we do in
    # local dev)
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        # use configured user credentials
        return storage.Client(project='voxsrc-2020-dev-1')
    else:
        # use embedded compute engine (GCP) credentials
        return storage.Client(
                credentials=compute_engine.Credentials(),
                project='voxsrc-2020-dev-1')

def upload_blob(bucket_name, dst_blob_name, src_file_name):
    storage_client = get_storage_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dst_blob_name)
    blob.upload_from_filename(src_file_name)

# @brief Download a blob from GCS
# @credit Google Cloud SDK docs
def download_blob(bucket_name, src_blob_name, dst_file_name):
    """Downloads a blob from the bucket."""
    storage_client = get_storage_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(src_blob_name)
    blob.download_to_filename(dst_file_name)

    print(f"Blob {src_blob_name} downloaded to {dst_file_name}.")

# @brief Downloads an individual file from a bucket. Uses gsutil for
#        parallel download (which has issues with credentials on the
#        cluster). More useful for local downloads. Doesn't require
#        python GCS SDK
def download_gcs_blob_in_parallel(src_bucket, src_file_path,
        dst_dir_path, num_cores=NUM_CORES_DEFAULT):
    start = time.time()
    src_url = f"gs://{src_bucket}/{src_file_path}"

    print(f"Downloading blob from {src_url} to directory {dst_dir_path}")

    dst_file_path = os.path.join(dst_dir_path, src_file_path.replace("/","-"))
    subprocess.call(f"gsutil \
                        -o 'GSUtil:parallel_thread_count=1' \
                        -o 'GSUtil:sliced_object_download_max_components={num_cores}' \
                        cp {src_url} {dst_file_path}", shell=True)
    print(f"...Finished in {time.time() - start} (s)")


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
        dst = os.path.join(args.save_tmp_data_to, blob)
        # @TODO get gsutil working in a docker container in order to
        #       perform parallel composite downloads, which apparently
        #       are not supported by the python client
        list_or_tar_name = dst
        extracted_data_name = dst.replace(".tar.gz","")
        if os.path.exists(list_or_tar_name) or \
                os.path.exists(extracted_data_name):
            print(f"Skipping pre-downloaded blob: {dst}")
        else:
            print(f"Downloading blob: {dst}")
            download_blob(args.data_bucket, blob, dst)
    print(f"...Finished in {time.time() - start} (s)")

# @brief New, better, dataset extractor. Takes an input path to a tar
#        file, and output path to extract to
# @note No tqdm since it doesn't work well on the cluster
# @param src_tar_path Full path to a .tar.gz file
# @param dst_extract_path Full path to the directory in which to place
#                         the extracted data (using the same name as
#                         in the .tar.gz)
def extract_tar(src_tar_path, dst_extract_path):
    start = time.time()
    dst_data_dir_name = os.path.basename(src_tar_path).replace('.tar.gz','')
    print(f"Extracting tar from {src_tar_path} to "
           "{os.path.join(dst_extract_path, dst_data_dir_name)}")

    with open(os.devnull, 'w') as FNULL:
        subprocess.call(f"tar -C {dst_extract_path} -zxvf {src_tar_path}",
                shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    print(f"...Finished in {time.time() - start} (s)")

# @note kept for compatibility with existing cluster data extraction
# @TODO change all consuming code to use extract_tar(...)
def extract_gcs_dataset(args):
    start = time.time()
    print(f"Uncompressing train/test data blobs...")

    data_blobs = [args.train_path, args.test_path]

    # uncompress data blobs
    for blob in data_blobs:
        src  = os.path.join(args.save_tmp_data_to, blob)
        dst = args.save_tmp_data_to
        with open(os.devnull, 'w') as FNULL:
            dst_dir_name = os.path.join(dst,
                    os.path.basename(src).replace(".tar.gz",""))
            if os.path.exists(dst_dir_name):
                print(f"Skipping extraction of file {src} into {dst_dir_name}")
            else:
                print(f"Extracting file {src} into {dst_dir_name}")
                subprocess.call(f"tar -C {dst} -zxvf {src}",
                        shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    print(f"...Finished in {time.time() - start} (s)")

# @brief Newer, better, version of transcode_gcs_dataset, but with a
#        different interface. Takes a directory containing AAC(.m4a)
#        files, and creates a .wav for every .m4a, but doesn't delete
#        the .m4a. .m4a files must be located at
#        {src_extract_path}/*/*/*.m4a, as they are in the train data
# @param src_extract_path The full path to the folder in which the
#                         extracted audio training data is
def convert_aac_to_wav(src_extract_path, save_tmp_data_to = "./"):
    start = time.time()

    print(f"Converting training data from AAC(.m4a)->WAV(.wav)...")

    # get list of all nested files
    files = glob.glob(os.path.join(src_extract_path, "*/*/*.m4a"))

    # @note Achieved best transcoding/audio-decompression results
    #       using GNU parallel, rather than multiple python processes
    # @note Use GNU Parallel 4.2; version 3 takes 3 times as long
    # @note Writing file names to file then cat and piping them back
    #       to parallel is weird... but seems to be efficient
    print("Writing wav file names to file")
    transcode_list_path = os.path.join(save_tmp_data_to,
            "transcode-list.txt")
    with open(transcode_list_path, "w") as outfile:
        outfile.write("\n".join(files))
    print("Constructing command")
    cmd = "cat %s | parallel ffmpeg -y -i {} -ac 1 -vn \
            -acodec pcm_s16le -ar 16000 {.}.wav >/dev/null \
            2>/dev/null" % (transcode_list_path)
    print("Uncompressing audio AAC->WAV in parallel...")
    subprocess.call(cmd, shell=True)

    print(f"...Finished in {time.time() - start} (s)")


# @TODO replace all uses of this with convert_aac_to_wav(...)
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

        if os.path.exists(files[0]):
            print("Skipping audio conversion of dataset; it "
                    "appears this has already been done")
        else:
            print(f"Proceeding with audio conversion of dataset")
            # @note Achieved best transcoding/audio-decompression results
            #       using GNU parallel, rather than multiple python processes
            # @note Use GNU Parallel 4.2; version 3 takes 3 times as long
            # @TODO speed this transcoding / audio compression stuff up more
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
                            for filename in files
                           ]

                print(f"Converting '{blob_dir_name}' files from AAC to WAV")
                pool = Pool(4) # num concurrent threads
                for i, returncode in tqdm(enumerate(pool.imap(partial(\
                        subprocess.call, shell=True), cmd_list))):
                    if returncode != 0:
                        print("%d command failed: %d" % (i, returncode))

    print(f"...Finished in {time.time() - start} (s)")

def set_loc_paths_from_gcs_dataset(args):
    # set new lists and data paths
    train_list = os.path.join(args.save_tmp_data_to, args.train_list)
    test_list = os.path.join(args.save_tmp_data_to, args.test_list)
    # @note remove the .tar.gz to reference extracted directories
    train_path = os.path.join(args.save_tmp_data_to,
            args.train_path.split(".tar.gz")[0])
    test_path = os.path.join(args.save_tmp_data_to,
            args.test_path.split(".tar.gz")[0])
    return train_list, test_list, train_path, test_path
