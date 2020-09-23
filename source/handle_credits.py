import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_credits(credit_filename="Credit.csv", production=False):
    # Load dataset, with the input predictors and output group
    credit = pd.read_csv(credit_filename)
    predictors = credit.iloc[:,0:20].values
    group = None
    if not production:
        group = credit.iloc[:,20].values
    
    return credit, predictors, group


def encode_credits(predictors, labelencoder=None):
    labelencoder = LabelEncoder() if labelencoder is None else labelencoder
    
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
    
    return predictors, labelencoder
