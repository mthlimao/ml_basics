from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Load iris dataset, and return unique targets and their quantities
iris = datasets.load_iris()
unique, quantity = np.unique(iris.target, return_counts = True)

# Creat clusters with KMeans
cluster = KMeans(n_clusters = 3)
cluster.fit(iris.data)

# Make confusion matrix
centroids = cluster.cluster_centers_
predictions = cluster.labels_

confusion = confusion_matrix(iris.target, predictions)

# Plot predicted clusters
plt.scatter(iris.data[predictions == 0, 0], iris.data[predictions == 0, 1], 
            c = 'green', label = 'Setosa')
plt.scatter(iris.data[predictions == 1, 0], iris.data[predictions == 1, 1], 
            c = 'red', label = 'Versicolor')
plt.scatter(iris.data[predictions == 2, 0], iris.data[predictions == 2, 1], 
            c = 'blue', label = 'Virgica')
plt.legend()