import os
import math
import torch
import numpy as np
from IterableTrainDataset import pad_spectrogram

# @brief Extracts n_subsets of overlapping desired_frames length. Each subset's offset
#        from the previous frame is dependent on its total length and the number
#        of subsets
def extract_eval_subsets_from_spectrogram(spectrogram, desired_frames,
        num_utterance_eval_subsets = 10):
    # pad if too small for network dims
    padded_spectrogram = pad_spectrogram(spectrogram, desired_frames)

    # make list of overlapping n_subsets start frames spanning the entire utterance
    actual_frames = spectrogram.shape[2]
    start_frames = np.linspace(0, actual_frames - desired_frames, num_utterance_eval_subsets)

    # append each utterance subset
    utterance_subsets = []
    for start_frame in start_frames:
        # @note appending the first element in first axis in order to get a list
        #       of NxM tensors, rather than 1xNxM
        utterance_subsets.append(padded_spectrogram[0,:,
            int(start_frame):int(start_frame)+desired_frames])

    # return stacked 3d tensor instead of list of 3d tensors
    return np.stack(utterance_subsets, axis=0)

# @brief provides an iterable yielding batches of features of all unique
#        utterances in the given test file (which contains utterance test pairs)
# @note this yields batches of all UNIQUE utterances so that consuming code can
#       cache forward pass output features (embeddings) that it computes. This
#       gives a substantial speedup since test pairs are all 2-combinations of
#       utterances in the test dataset, and therefore contain many many
#       duplicate utterances
class IterableEvalDataset(torch.utils.data.IterableDataset):
    def __init__(self, test_list_path, test_data_path, num_desired_frames,
            num_utterance_eval_subsets, batch_size):
        super(IterableEvalDataset).__init__()
        self.test_list_path = test_list_path
        self.test_data_path = test_data_path
        self.num_desired_frames = num_desired_frames
        self.num_utterance_eval_subsets = num_utterance_eval_subsets
        self.batch_size = batch_size

        # read the utterance list file to store unique paths
        nonunique_utterance_test_paths = []
        with open(test_list_path) as utterance_test_pairs:
            line = utterance_test_pairs.readline()
            while line:
                data = line.split()
                # add both utterances in pair to list
                nonunique_utterance_test_paths.append(data[1])
                nonunique_utterance_test_paths.append(data[2])
                line = utterance_test_pairs.readline()
        self.unique_utterance_test_paths = list(set(nonunique_utterance_test_paths))

        self.batch_indices = range(math.ceil(len(self.unique_utterance_test_paths)
            / self.batch_size))

    def __len__(self):
        return len(self.batch_indices)

    # @return Iterator for a worker's subset of the dataset
    # @TODO use pytorch's batching, rather than manual
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        print(f"IterableEvalDataset: Starting worker thread #{worker_info.id}")
        worker_batch_indices = self.batch_indices[worker_info.id :: worker_info.num_workers]

        for batch_index in worker_batch_indices:
            batch_sample_features = []
            # get indices of samples in batch
            first_sample_index = batch_index * self.batch_size
            last_sample_index_non_inclusive = min(first_sample_index
                    + self.batch_size, len(self.unique_utterance_test_paths))
            # save the original utterance paths for lookup in consuming code
            # when caching embeddings produced by each set of utterance subsets
            batch_utterance_paths = self.unique_utterance_test_paths[
                    first_sample_index : last_sample_index_non_inclusive]

            # load and append each sample's features
            for sample_utterance_path in batch_utterance_paths:
                # extract Subsets x Freq x Frames tensor from a single utterance in
                full_path = os.path.join(self.test_data_path,
                        sample_utterance_path).replace(".wav", ".npy")
                # load raw spectrogram features
                full_utterance_spectrogram = np.load(full_path).astype(np.float16)

                # convert to torch tensors of multiple subsets of this utterance
                # from start to finish
                overlapping_spectrogram_subsets = torch.from_numpy(
                        extract_eval_subsets_from_spectrogram(
                        full_utterance_spectrogram, self.num_desired_frames,
                        self.num_utterance_eval_subsets))
                overlapping_spectrogram_subsets.requires_grad = False
                batch_sample_features.append(overlapping_spectrogram_subsets)

            # pack sample features in proper tensor dims
            # of (batch_size * n_subsets) x (freq_bins) x (subset_time_length)
            batch = torch.cat(batch_sample_features, dim=0)

            yield [batch_utterance_paths, batch]

        print(f"IterableEvalDataset: Stopping worker #{worker_info.id}")
