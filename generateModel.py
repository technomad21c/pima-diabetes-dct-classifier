# source: https://www.datacamp.com/community/tutorials/decision-tree-classification-python

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv("data/diabetes.csv", header=None, names=col_names, skiprows=1)
data.head()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
x = data[feature_cols]
y = data.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# from sklearn import tree
# import pydotplus
# import pydot
# from io import StringIO
#
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = feature_cols,class_names=['0','1'])
#
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#
# graph.write_png("tree.png")

# persist the model
from joblib import dump
dump(clf, 'pima-diabetes.joblib')
