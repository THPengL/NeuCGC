
import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering
from utils import cosine_distance, euclidean_distance



def initialize(X, n_clusters):
    """
    Initialize cluster centers.

    Parameters
    - X: (torch.tensor) matrix
    - n_clusters: (int) number of clusters
    
    Return
    - initial state: (np.array) 
    """
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    initial_state = X[indices]

    return initial_state


def k_means(X, n_clusters, distance='euclidean', tol=1e-4, device=torch.device('cuda:0')):
    """
    Perform k-means algorithm on X.

    Parameters
    - X: torch.tensor. matrix
    - n_clusters: int. number of clusters
    - distance: str. pairwise distance 'euclidean'(default) or 'cosine'
    - tol: float. Threshold 
    - device: torch.device. Running device
    
    Return
    - choice_cluster: torch.tensor. Predicted cluster ids.
    - initial_state: torch.tensor. Predicted cluster centers.
    - dis: minimum pair wise distance.
    """
    if distance == 'euclidean':
        pairwise_distance_function = euclidean_distance
    elif distance == 'cosine':
        pairwise_distance_function = cosine_distance
    else:
        raise NotImplementedError(f"Not implemented '{distance}' distance!")

    X = X.float()
    X = X.to(device)

    # initialize
    dis_min = float('inf')
    initial_state_best = initialize(X, n_clusters)
    # initial_state_best = None
    for i in range(20):
        initial_state = initialize(X, n_clusters)
        dis = pairwise_distance_function(X, initial_state).sum()

        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state

    initial_state = initial_state_best

    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if iteration > 500:
            break
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), initial_state, dis


def clustering(feature, cluster_num, seed = 10, method="kmeans", device = torch.device('cpu')):
    """
    Clustering using feature matrix.

    Parameters
    - feature: feature matrix.
    - cluster_num: number of clusters.
    - seed: random seed.
    - device: torch.device. device where the clustering algorithm will be running on.

    Return
    - label_pred: predicted label.
    """
    if method == 'kmeans':
        if device == torch.device('cuda'):
            label_pred, centers, dis = k_means(X=feature,
                                        n_clusters=cluster_num,
                                        distance="cosine",           # cosine euclidean
                                        device=device)
            label_pred = label_pred.numpy()
            return label_pred, centers, dis
        else:
            feature = feature.cpu().numpy()
            kmeans = KMeans(n_clusters=cluster_num,
                            random_state=seed,
                            init='k-means++')
            label_pred = kmeans.fit_predict(feature)
            centers = kmeans.cluster_centers_
            return label_pred
    elif method == 'spectral':
        feature = feature.cpu().numpy()
        spectral = SpectralClustering(gamma=1.0,
                                    n_clusters=cluster_num,
                                    random_state=seed)
        label_pred = spectral.fit_predict(feature)
        return label_pred
