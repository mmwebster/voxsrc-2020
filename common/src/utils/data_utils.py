from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import subprocess
import time
import glob
import os
from google.cloud import storage
from google.auth import compute_engine
from pathlib import Path

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

def upload_blob(bucket_name, dst_blob_path, src_file_path):
    start_time = time.time()
    storage_client = get_storage_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dst_blob_path)

    blob.upload_from_filename(src_file_path)

    print(f"Uploaded {src_file_path} to gcs://{bucket_name}/{dst_blob_path} "
          f"in {time.time() - start_time} (s)")

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
def download_gcs_dataset(bucket, save_path, blobs):
    start = time.time()
    print("Downloading dataset blobs...")

    # download each blob
    for blob in blobs:
        dst = os.path.join(save_path, blob)
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
            download_blob(bucket, blob, dst)
    print(f"...Finished in {time.time() - start} (s)")

# @brief New, better, dataset extractor. Takes an input path to a tar
#        file, and output path to extract to
# @note No tqdm since it doesn't work well on the cluster
# @param src_tar_path Full path to a .tar.gz file
# @param dst_extract_path Full path to the directory in which to place
#                         the extracted data (using the same name as
#                         in the .tar.gz)
def extract_tar(src_tar_path, dst_extract_path, use_pigz=False):
    start = time.time()
    dst_data_dir_name = os.path.basename(src_tar_path).replace('.tar.gz','')
    print(f"Extracting tar from {src_tar_path} to "
           "{os.path.join(dst_extract_path, dst_data_dir_name)}")

    with open(os.devnull, 'w') as FNULL:
        if use_pigz:
            subprocess.call(f"tar -C {dst_extract_path} -I pigz -xf {src_tar_path}",
                    shell=True)
        else:
            subprocess.call(f"tar -C {dst_extract_path} -zxf {src_tar_path}",
                    shell=True)

    print(f"...Finished in {time.time() - start} (s)")

# @note kept for compatibility with existing cluster data extraction
# @TODO change all consuming code to use extract_tar(...)
def extract_gcs_dataset(args, use_pigz=False):
    start = time.time()
    print(f"Uncompressing train/test data blobs...")

    data_blobs = [args.train_path, args.test_path]
    # hacky thing to only apply pigz to train data and not test. Unnecessary
    # once test's features are also pre-extracted and the dataset is compressed
    # with pigz
    uses_pigz = [use_pigz, False]

    # uncompress data blobs
    for index, blob in enumerate(data_blobs):
        src  = os.path.join(args.save_tmp_data_to, blob)
        dst = args.save_tmp_data_to
        with open(os.devnull, 'w') as FNULL:
            dst_dir_name = os.path.join(dst,
                    os.path.basename(src).replace(".tar.gz",""))
            if os.path.exists(dst_dir_name):
                print(f"Skipping extraction of file {src} into {dst_dir_name}")
            else:
                print(f"Extracting file {src} into {dst_dir_name}")
                if uses_pigz[index]:
                    subprocess.call(f"tar -C {dst} -I pigz -xf {src}", shell=True)
                else:
                    subprocess.call(f"tar -C {dst} -zxf {src}", shell=True)

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
        files_to_convert = glob.glob(f"{blob_dir_path}/*/*/*.m4a")
        converted_files = glob.glob(f"{blob_dir_path}/*/*/*.wav")

        if len(converted_files) > 0:
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
                    outfile.write("\n".join(files_to_convert))
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
                            for filename in files_to_convert
                           ]

                print(f"Converting '{blob_dir_name}' files from AAC to WAV")
                pool = Pool(4) # num concurrent threads
                for i, returncode in enumerate(pool.imap(partial(\
                        subprocess.call, shell=True), cmd_list)):
                    if returncode != 0:
                        print("%d command failed: %d" % (i, returncode))

    print(f"...Finished in {time.time() - start} (s)")

# @brief Returns local paths to downloaded data, in the same order as recieved.
#        Just prepends data_path to all blobs and removes the tar.gz
def get_loc_paths_from_gcs_dataset(save_path, blobs):
    out_paths = []
    for blob in blobs:
        no_tar = blob.split(".tar.gz")[0]
        out_paths.append(os.path.join(save_path, no_tar))
    return out_paths

# @brief Compress a directory into a tar file. Usage includes compressing
#        extracted feature directory
# @param src_dir_path Full path to directory to compress, with or without a
#                     trailing slash
# @param dst_file_path Full path to tar file output, including the
#                      trailing .tar.gz
def compress_to_tar(src_dir_path, dst_file_path, use_pigz=False):
    # add trailing slashes if not present
    start_time = time.time()
    src_dir_path = os.path.join(src_dir_path, '')

    # ensure tar parent dir exists, and extract other paths
    Path(os.path.dirname(dst_file_path)).mkdir(parents=True, exist_ok=True)
    src_dir_split = os.path.split(os.path.dirname(src_dir_path))
    src_dir_parent_path = src_dir_split[0]
    src_dir_name = src_dir_split[1]

    # tar it
    cmd = None
    if use_pigz:
        cmd = f"tar -C {src_dir_parent_path} -I pigz -cf {dst_file_path} {src_dir_name}"
    else:
        cmd = f"tar -C {src_dir_parent_path} -zcf {dst_file_path} {src_dir_name}"
    subprocess.call(cmd, shell=True)

    print(f"Compressed {src_dir_path} to {dst_file_path} in "
          f"{time.time() - start_time} (s)")
