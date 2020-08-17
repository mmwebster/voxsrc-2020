import threading
from pathlib import Path
import queue
from scipy.io import wavfile
import numpy
import os
import time
import torch
import subprocess
import sys

class FeatureExtractor():
    def __init__(self, src_list_path, src_data_path, dst_feats_path,
            feature_extractor_fn, num_threads = 10, job_max_queue_size = 50):
        # input params
        self.src_list_path = src_list_path
        self.src_data_path = src_data_path
        self.dst_feats_path = dst_feats_path
        self.feature_extractor_fn = feature_extractor_fn
        self.num_threads = num_threads

        # misc
        update_interval_percent = 10
        self.num_jobs = self.get_num_lines(self.src_list_path)
        self.update_interval = int(self.num_jobs / update_interval_percent)
        self.threads = []
        self.job_queue = queue.Queue(job_max_queue_size)
        self.done = False
        self.error = False

    # start up threads on entering context
    def __enter__(self):
        self.spawn_threads()
        return self

    # stop all threads on exiting context
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.join_threads()
        print(f"FeatureExtractor: Joined all {self.num_threads} worker threads")
        return True

    # @brief begins synchronously queueing jobs for feature extraction
    # @TODO accept an X param and give timing updates every X% through
    #       processing the dataset
    # @TODO make async and add a join method
    def run(self):
        lines_processed = 0
        with open(self.src_list_path) as src_list:
            print(f"FeatureExtractor: Beginning to queue file paths")
            while not self.error:
                # grab a line in file
                line = src_list.readline()
                if not line:
                    break
                # extract the path to the utterance wav
                utterance_wav_path_from_data_root = line.split()[1]
                # queue it as a job
                failed_to_queue = True
                while failed_to_queue and not self.error:
                    try:
                        self.job_queue.put(utterance_wav_path_from_data_root,
                                timeout=.1)
                        failed_to_queue = False
                    except queue.Full:
                        print(f"FeatureExtractor: Timed out while queuing job")
                lines_processed += 1
                if lines_processed % self.update_interval == 0:
                    print(f"FeatureExtractor: Queued "
                          f"{int(100*lines_processed/self.num_jobs)}% of "
                          f"{self.num_jobs} jobs")
            print(f"FeatureExtractor: Finished queueing file paths")

    def spawn_threads(self):
        print(f"FeatureExtractor: Spawning {self.num_threads} worker threads")
        for thread_index in range(0, self.num_threads):
            self.threads.append(threading.Thread(
                target = self.feature_extractor_thread,
                args = [thread_index]))
            self.threads[-1].start()

    def join_threads(self):
        print("FeatureExtractor: Joining threads")
        self.done = True
        # wait for threads to finish remaining items in queue
        for index, thread in enumerate(self.threads):
            thread.join()

    def feature_extractor_thread(self, thread_index):
        print(f"FeatureExtractor: Thread #{thread_index} starting")
        while not self.error and not (self.done and self.job_queue.empty()):
            try:
                # grab a job (path to utterance to proc), waiting if unavailable
                utterance_wav_path_from_root = self.job_queue.get(timeout=.1)
                utterance_wav_full_path = os.path.join(self.src_data_path,
                        utterance_wav_path_from_root)
                # read
                utterance_wav = self.load_wav_from_file(utterance_wav_full_path)
                # transform
                utterance_feats = self.extract_features_from_wav(utterance_wav)
                # write
                utterance_feats_path = os.path.join(self.dst_feats_path,
                        utterance_wav_path_from_root).replace(".wav", "")
                self.write_features_to_file(utterance_feats, utterance_feats_path)
            except queue.Empty:
                print(f"FeatureExtractor: Thread #{thread_index} timed out while"
                       " waiting for an utterance")
            except FileNotFoundError:
                print("FeatureExtractor: Utterance file not found. Killing "
                      "workers. Did tar extraction fail?")
                self.error = True
            except:
                print(f"FeatureExtractor: Unexpected exception. Killing workers. "
                      f"Error: {sys.exc_info()[0]}")
                self.error = True
                raise
        print(f"FeatureExtractor: Thread #{thread_index} stopping")

    # @TODO Need the torch tensor?
    def load_wav_from_file(self, utterance_wav_path):
        sample_rate, audio  = wavfile.read(utterance_wav_path)
        feats = []
        feats.append(audio)
        feat = numpy.stack(feats, axis=0)
        feat = torch.FloatTensor(feat)
        return feat

    # @brief run user-defined feature extraction
    # @note user defined function must return a numpy array
    def extract_features_from_wav(self, utterance_wav):
        return self.feature_extractor_fn(utterance_wav)

    # @brief write numpy array to file
    def write_features_to_file(self, utterance_feats, utterance_feats_path):
        # ensure path to file exists
        Path(os.path.dirname(utterance_feats_path)).mkdir(parents=True,
                exist_ok=True)
        # save the features to file
        numpy.save(utterance_feats_path, utterance_feats)

    # @TODO move to utils
    def get_num_lines(self, file_path):
        wc_cmd = subprocess.check_output(['wc', '-l', file_path])
        return int(wc_cmd.decode('utf8').split()[0])
