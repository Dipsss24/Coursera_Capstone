import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# generation data to use k-means clustering.
np.random.seed(0)  # setting the seed so that every time the same random numbers are calculated.

# make random clusters of points using make_blobs class.
# x stores the coordinates of the points.
# y stores which cluster does each coordinate belong to.
x, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.show()
plt.close()

# setting up k-means
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)  # selects initial cluster centers for k-means clustering
# in a smart way, n_clusters is the number of clusters and centroids, n_init is the number of times the algo will run
# with different centroid seeds.
k_means.fit(x)  # fitting the k-means model.

k_means_label = k_means.labels_  # gives the cluster number for each data point
k_means_centers = k_means.cluster_centers_  # gives the centroids

# plot the results
# colors uses a color map, which will produce an array of colors based on the number of labels. We use
# set(k_means_labels) to get the unique labels.
fig = plt.figure(figsize=(15, 10))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_label))))

ax = fig.add_subplot(1, 1, 1)

# loop to plot data points with colors
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = (k_means_label == k)  # list of true and false to know which data point belongs
    cluster_center = k_means_centers[k]  # get the centroid for the current cluster
    ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor=col, marker='.')  # plot data points with color
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()
plt.close()
