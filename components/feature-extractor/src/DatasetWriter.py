import threading
import queue
from pathlib import Path
import numpy
import os
import time

class DatasetWriter():
    def __init__(self, max_queue_size = 50, num_threads = 30):
        # init members
        self.queue = queue.Queue(max_queue_size)
        self.queue_cv = threading.Condition()
        self.num_threads = num_threads
        self.done = False
        self.threads = []

    def __enter__(self):
        self.spawn_threads()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.join_threads()
        return True

    # @brief Enqueue an utterance for writing to file system, will block if the
    #        queue is full
    def enqueue(self, utterance_feats, utterance_path):
        # enqueue the item, blocking until queue not full
        self.queue.put([utterance_feats, utterance_path])
        print(f"Enqueued utterance for writing")

    def data_writer_thread(self, thread_index):
        print(f"consumer #{thread_index} starting")
        while not self.done or not self.queue.empty():
            # dequeue an item, blocking if none are available (until timeout)
            start = time.time()
            try:
                utterance_feats, utterance_feats_path = self.queue.get(timeout=.1)
                # ensure path to file exists
                Path(os.path.dirname(utterance_feats_path)).mkdir(parents=True,
                        exist_ok=True)
                # save the features to file
                numpy.save(utterance_feats_path, utterance_feats)
                print(f"consumer #{thread_index} wrote data to file")
            except queue.Empty:
                print(f"consumer #{thread_index} timed out on queue.get() after {time.time() - start}")
        print(f"consumer #{thread_index} stopping")

    def spawn_threads(self):
        for thread_index in range(0, self.num_threads):
            self.threads.append(threading.Thread(target = self.data_writer_thread,
                args = [thread_index]))
            self.threads[-1].start()

    def join_threads(self):
        print("Joining threads")
        self.done = True
        # wait for threads to finish remaining items in queue
        for index, thread in enumerate(self.threads):
            print(f"Joining thread #{index}")
            thread.join()
