# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array


class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
            """
            This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
            will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
            this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
            the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
            and/or encapsulate the necessary mechanisms in additional functions.
            :param X: Array that contains the input feature vectors
            :param y: Unused
            :return: Returns the clustering object itself.
            """
            X = check_array(X, accept_sparse='csr')
            
            if self.cluster_centers_ is None:
                self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]

            if self.labels_ is None:
                self.labels_ = np.zeros(X.shape[0])

            for iter in range(self.max_iter):
                euclidean_distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_, axis=2)
                new_labels = np.argmin(euclidean_distances, axis=1)

                new_centers = []
                for i in range(self.n_clusters):
                    cluster_points = X[new_labels == i]
                    if len(cluster_points) > 0:
                        mean_center = np.mean(cluster_points, axis=0)
                        new_centers.append(mean_center)
                    else:
                        new_centers.append(self.cluster_centers_[i])
                new_centers = np.array(new_centers) if new_centers else self.cluster_centers_

                if np.array_equal(new_labels, self.labels_) and np.allclose(self.cluster_centers_, new_centers):
                    break

                self.labels_ = new_labels
                self.cluster_centers_ = new_centers

            return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances (optional).
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

  
    def fit(self, X: np.ndarray, y=None):
        X = check_array(X, accept_sparse='csr')
        
        self.labels_ = np.full(X.shape[0], fill_value=-1)  

        cluster_id = 0  

        for i in range(X.shape[0]):
            if self.labels_[i] != -1:  
                continue

            neighbors = self.get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  
            else:
                self.expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

        return self
    def expand_cluster(self, X, start_object, neighbors, cluster_id):
        self.labels_[start_object] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.labels_[neighbor] == -1:
                self.labels_[neighbor] = cluster_id
                new_neighbors = self.get_neighbors(X, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            i += 1

    def get_neighbors(self, X, id):
    
        X_float = X.astype(np.float32)
        euclidean_distances = np.linalg.norm(X_float - X_float[id, np.newaxis], axis=1)
        return np.where(euclidean_distances <= self.eps)[0]

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
