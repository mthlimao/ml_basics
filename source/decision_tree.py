import graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from handle_credits import load_credits, encode_credits


# Load dataset, with the input predictors and output group
credit, predictors, group = load_credits()

# Convert non numeric inputs to numeric inputs by LabelEncoder
predictors, labelencoder = encode_credits(predictors)


# Fit naive bayes model
X_train, X_test, y_train, y_test = train_test_split(predictors,
                                                    group,
                                                    test_size = 0.3,
                                                    random_state = 0)
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)


# Make predictions with the fitted model based on test data
predictions = tree.predict(X_test)
confusion = confusion_matrix(y_test, predictions)
acc_rate = accuracy_score(y_test, predictions)
error_rate = 1 - acc_rate

export_graphviz(tree, out_file = 'tree.dot')

# Predict given a new unseen production case
_, new_credit, _ = load_credits('NewCredit.csv', production=True)
new_credit, labelencoder = encode_credits(new_credit, labelencoder)

new_prediction = tree.predict(new_credit)
