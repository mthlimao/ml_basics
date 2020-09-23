import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset, with the input predictors and output group
credit = pd.read_csv('Credit.csv')
predictors = credit.iloc[:,0:20].values
group = credit.iloc[:,20].values


