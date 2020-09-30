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

# Generate Confusion Matrix
lst_predictions = []
lst_real = []
for i in range(len(predictions)):
    print('----')
    print(i)
    print('----')
    for j in range(len(predictions[i])):
        lst_predictions.append(i)
        lst_real.append(iris.target[predictions[i][j]])
        lst_predictions
lst_predictions = np.asarray(lst_predictions)
lst_real = np.asarray(lst_real)
results = confusion_matrix(lst_real, lst_predictions)
