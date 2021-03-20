from joblib import load
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv("data/diabetes.csv", header=None, names=col_names, skiprows=1)
data.head()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
x = data[feature_cols]
y = data.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# load the trained model
clf = load('pima-diabetes.joblib')
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))