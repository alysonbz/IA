from src.utils import load_points
import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def assign_cluster(points, centroids):
    labels = []
    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_label = np.argmin(distances)
        labels.append(cluster_label)
    return labels

def update_centroids(points, labels, n_clusters):
    centroids = []
    for cluster_label in range(n_clusters):
        cluster_points = points[labels == cluster_label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

points = load_points()

test_points = points[:50,:]
train_points = points[50:,:]

# Initialize centroids randomly
np.random.seed(0)
n_clusters = 3
centroids = np.random.randn(n_clusters, train_points.shape[1])

labels = assign_cluster(test_points, centroids)

print(labels)

xs = test_points[:,0]
ys = test_points[:,1]

plt.scatter(xs, ys, c=labels, alpha=0.5)

centroids = update_centroids(train_points, labels, n_clusters)

centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(centroids_x, centroids_y, s=50, marker='D')
plt.show()
