import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix
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
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)


# Make predictions with the fitted model based on test data
predictions = naive_bayes.predict(X_test)
confusion = confusion_matrix(y_test, predictions)
acc_rate = accuracy_score(y_test, predictions)
error_rate = 1 - acc_rate

# Improve visual of confusion matrix with yellow brick library
v = ConfusionMatrix(GaussianNB())
v.fit(X_train, y_train)
v.score(X_test, y_test)
v.poof()

# Predict given a new unseen production case
_, new_credit, _ = load_credits('NewCredit.csv', production=True)
new_credit, labelencoder = encode_credits(new_credit, labelencoder)

new_prediction = naive_bayes.predict(new_credit)
