"""Get embeddings & plot with T-SNE or UMAP."""
import os
import sys
import torch
import manifold

sys.path.insert(0, '../../components/train/src')
from SpeakerNet import SpeakerNet
from DatasetLoader import loadWAV

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")


def parse_test_list(test_list):
    """ Returns all unique WAV filepaths from an eval file
    
    Example file contents
    1 path/to/speaker1_a.wav path/to/speaker2_a.wav
    0 path/to/speaker1_b.wav path/to/speaker2_b.wav
    1 path/to/speaker1_c.wav path/to/speaker2_c.wav

    Parameters
    ----------
    test_list : str
        The test_list file path.

    Returns
    -------
    wav_paths : list
        List of str. Unique wav file paths found in this file.
    """
    files = []

    with open(test_list) as listfile:
        for line in listfile:
            _, speaker1_wav, speaker2_wav = line.split();
            files.append(speaker1_wav)
            files.append(speaker2_wav)

    wav_paths = list(set(files))
    wav_paths.sort()
    return wav_paths

# TODO(alexbooth): Actually would be really helpful if all our models just implemented an API like this 
def get_embeddings(model, test_list, test_path, max_frames=1000):
    """2D scatter plot
    
    Parameters
    ----------
    model : torch.nn.Module
        The loaded model.
        
    test_list : str
        The list of wav file names.
        
    test_path : str
        The wav file location.
        
    max_frames : int
        The max number of frames to process per utterance
        
    Returns
    -------
    embeddings : numpy.ndarray
        Stacked speaker embeddings per utterance. Each row is an embedding.
        
    labels : list
        List of str. Corresponding speaker IDs.
    """
    # TODO(alexbooth): what does num_eval do?
    num_eval = 10

    wav_paths = parse_test_list(test_list)
    
    embeddings = None 
    labels = []
    
    for idx, wav_path in enumerate(wav_paths):
        input_ = loadWAV(os.path.join(test_path, wav_path), max_frames, evalmode=True, num_eval=num_eval).to(device)
        output = model.forward(input_).detach().cpu()
        
        if embeddings == None:
            embeddings = output.view(-1).unsqueeze(0)
        else:
            output = output.view(-1).unsqueeze(0)
            embeddings = torch.cat([embeddings, output])
            
        labels.append(wav_path.split('/')[0])
        
    return embeddings, labels


if __name__ == '__main__':
    """ Usage:
    python3 visualize.py  --projection 3d --algorithm umap                         \
                          -m ../../components/train/tmp/model/model000000001.pt    \
                          --test_list ../../data/lists/vox1_no_cuda.txt            \
                          --test_path ../../components/train/tmp/data/vox1_no_cuda
    """
    import argparse
    parser = argparse.ArgumentParser();
    parser.add_argument('-p', '--projection', default="2d", choices=['2d', '3d'], 
                        dest="projection", action="store",  type=str)
    parser.add_argument('-m', '--model', help="Path to model",
                        dest="model_path", action="store",  type=str, required=True)
    parser.add_argument('-a', '--algorithm', default="tsne", choices=['tsne', 'umap'],
                        dest="algorithm", action="store",  type=str)
    parser.add_argument('-n', '--num_frames', default=1000, help="Max number of frames per utterance",
                        dest="num_frames", action="store",  type=int)
    parser.add_argument('-s', '--seed', default=0, help="RNG seed for initial state",
                        dest="seed", action="store",  type=int)
    parser.add_argument('--test_list', required=True, help="Test list",
                        dest="test_list", action="store",  type=str)
    parser.add_argument('--test_path', required=True, help="Test path",
                        dest="test_path", action="store",  type=str)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise Exception("Invalid model path")

    model = torch.load(args.model_path)
    model.eval()
    
    X, y = get_embeddings(model.__S__, args.test_list, args.test_path, max_frames=args.num_frames)

    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    tsne_config = {
        "n_components": 2 if args.projection == '2d' else 3, 
        "random_state": args.seed,
    }  
    
    # https://umap.scikit-tda.org/
    umap_config = {
        "n_components": 2 if args.projection == '2d' else 3, 
        "random_state": args.seed,
    }
        
    if args.algorithm == "tsne":
        X_embedded = manifold.compute_tsne(X, **tsne_config)
    if args.algorithm == "umap":
        X_embedded = manifold.compute_umap(X, **umap_config)
        
    if args.projection == "2d":
        manifold.plot2d(X_embedded, y)
    elif args.projection == "3d":
        manifold.plot3d(X_embedded, y)
