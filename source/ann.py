import time
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Fix random seed for reproducibility
seed = 0
np.random.seed(seed)

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Define model
def simple_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define hyperparameter to be optimized, estimator, pipeline and grid
epochs = [100, 200, 500]

estimator = KerasClassifier(build_fn=simple_model, batch_size=5, verbose=0)
pipeline = Pipeline([('standardize', StandardScaler()), ('mlp', estimator)])
param_grid = dict(mlp__epochs=epochs)

grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10)
results = grid.fit(X, dummy_y)

print(f"Best parameters for simple_model:")
print(f"{results.best_params_}")
print()

means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']

print("Grid scores obtained: ")
for mean, std, params in zip(means, stds, results.cv_results_['params']):
    print(f"{mean:.4f} (+/-{std:.4f}) using {params}")
print()
