import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KMeans:
    def __init__(self, K=5, MAX_ITER=100):
        self.K = K
        self.MAX_ITER = MAX_ITER

        self.clusters = None
        self.centroids = None

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = self.X[random_sample_idxs]

        for _ in range(self.MAX_ITER):
            self.clusters = self.create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)

            if self.is_converged(centroids_old, self.centroids):
                break

        return self.get_cluster_labels(self.clusters)

    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        return np.argmin(distances)

    def get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            centroids[cluster_idx] = np.mean(self.X[cluster], axis=0)
        return centroids

    def is_converged(self, old, new):
        distances = [euclidean_distance(old[i], new[i]) for i in range(self.K)]
        return sum(distances) == 0

    def get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def inertia(self):
        total = 0
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                total += euclidean_distance(
                    self.X[sample_idx], self.centroids[cluster_idx]
                ) ** 2
        return total
