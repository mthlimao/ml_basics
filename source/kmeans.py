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
