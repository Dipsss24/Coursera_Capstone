import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

customer_df = pd.read_csv('Cust_Segmentation.csv')
df = customer_df.drop('Address', axis=1)  # drop the address column

X = df.values[:, 1:]  # take all rows and all columns except the customer id
X = np.nan_to_num(X)  # handle NaN values using numpy function
cluster_dataset = StandardScaler().fit_transform(X)  # Use Standard scaler to normalize the dataset

num_cluster = 3  # decide the number of clusters needed for customers.
k_means = KMeans(init='k-means++', n_clusters=num_cluster, n_init=12)  # initialise
k_means.fit(cluster_dataset)
labels = k_means.labels_  # get labels for the clustered model.

df['labels'] = labels  # add a column to the data frame
cluster_centroid = df.groupby('labels').mean()  # get centroids by grouping by the labels
