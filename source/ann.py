import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Fix random seed for reproducibility
seed = 0
numpy.random.seed(seed)

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target
