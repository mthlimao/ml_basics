from sklearn import datasets
from sklearn.metrics import confusion_matrix
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

# Load data and create clusters with KMedoid
iris = datasets.load_iris()

cluster = kmedoids(iris.data[:, 0:2], [3, 12, 20])
cluster.get_medoids()
cluster.process()
predictions = cluster.get_clusters()
medoids = cluster.get_medoids()

# Visualize Medoids
v = cluster_visualizer()
v.append_clusters(predictions, iris.data[:,0:2])
v.append_cluster(medoids, data = iris.data[:,0:2], marker = '*', markersize = 15)
v.show()
