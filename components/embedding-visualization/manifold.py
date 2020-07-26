"""Wrapper around UMAP & TSNE for embedding visualizations."""
import umap
from sklearn.manifold import TSNE

import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_tsne(X, **kwargs):
    """Compute T-SNE embeddings of a dataset
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    
    Parameters
    ----------
    X : numpy.ndarray
        The input dataset.
        
    kwargs : dict
        T-SNE config (see sklearn docs).
        
    Returns
    -------
    X_embedding : numpy.ndarray
        The T-SNE embedding.
    """
    return TSNE(**kwargs).fit_transform(X)
    
def compute_umap(X, **kwargs):
    """Compute UMAP embeddings of a dataset
    https://umap.scikit-tda.org/
    
    Parameters
    ----------
    X : numpy.ndarray
        The input dataset.
        
    kwargs : dict
        UMAP config (see umap docs).
                
    Returns
    -------
    X_embedding : numpy.ndarray
        The UMAP embedding.
    """
    return umap.UMAP(**kwargs).fit_transform(X)

def plot2d(X, y):
    """2D scatter plot
    
    Parameters
    ----------
    X : numpy.ndarray
        The input dataset.
        
    y : numpy.ndarray
        The corresponding labels.
    """
    plt.figure()
    sns.scatterplot(
        X[:,0], 
        X[:,1],
        hue=y,
        palette=sns.color_palette("colorblind", n_colors=len(set(y))),
        alpha=0.3
    )
    plt.show()

def plot3d(X, y):
    """3D scatter plot
    
    Parameters
    ----------
    X : numpy.ndarray
        The input dataset.
        
    y : numpy.ndarray
        The corresponding labels.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X[:,0],
        X[:,1],
        X[:,2],
        c=[hash(label) for label in y],
        cmap=cm.get_cmap('viridis', len(set(y))),
        alpha=0.3
    )
    plt.show()
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser();
    parser.add_argument('-p', '--projection', default="2d", choices=['2d', '3d'], 
                        dest="projection", action="store",  type=str)
    parser.add_argument('-a', '--algorithm', default="tsne", choices=['tsne', 'umap'],
                        dest="algorithm", action="store",  type=str)
    parser.add_argument('-s', '--seed', default=0, help="RNG seed for initial state",
                        dest="seed", action="store",  type=int)
    args = parser.parse_args()

    # Test on mnist
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    
    config = {
        "n_components": 2 if args.projection == '2d' else 3, 
        "random_state": args.seed
    }

    if args.algorithm == "tsne":
        X_embedded = compute_tsne(X, **config)
    elif args.algorithm == "umap":
        X_embedded = compute_umap(X, **config)
        
    if args.projection == "2d":
        plot2d(X_embedded, y)
    elif args.projection == "3d":
        plot3d(X_embedded, y)
