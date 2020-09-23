import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix


# Load dataset, with the input predictors and output group
credit = pd.read_csv('Credit.csv')
predictors = credit.iloc[:,0:20].values
group = credit.iloc[:,20].values

# Convert non numeric inputs to numeric inputs by LabelEncoder
labelencoder = LabelEncoder()
predictors[:,0] = labelencoder.fit_transform(predictors[:,0])
predictors[:,2] = labelencoder.fit_transform(predictors[:,2])
predictors[:, 3] = labelencoder.fit_transform(predictors[:, 3])
predictors[:, 5] = labelencoder.fit_transform(predictors[:, 5])
predictors[:, 6] = labelencoder.fit_transform(predictors[:, 6])
predictors[:, 8] = labelencoder.fit_transform(predictors[:, 8])
predictors[:, 9] = labelencoder.fit_transform(predictors[:, 9])
predictors[:, 11] = labelencoder.fit_transform(predictors[:, 11])
predictors[:, 13] = labelencoder.fit_transform(predictors[:, 13])
predictors[:, 14] = labelencoder.fit_transform(predictors[:, 14])
predictors[:, 16] = labelencoder.fit_transform(predictors[:, 16])
predictors[:, 18] = labelencoder.fit_transform(predictors[:, 18])
predictors[:, 19] = labelencoder.fit_transform(predictors[:, 19])


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

