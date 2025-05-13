import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import accuracy_score

df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')
df.shape
# removing duplications
df_duplicate = df[df.duplicated(keep = False)]
df_duplicate.head(10)
print(df.isnull().sum())
# dropping null values
copyDF = copy.deepcopy(df)
copyDF = copyDF.dropna(axis=0)
npdf[0]

# encoding categorical Variables
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(copyDF['Gender'])
#print(type(le.transform(copyDF['Gender'])))
le.fit(copyDF['Result'])
#print(le.transform(copyDF['Result']))
le.fit(copyDF['Gender'])
copyDF['Gender'] = le.transform(copyDF['Gender'])
le.fit(copyDF['Result'])
copyDF['Result'] = le.transform(copyDF['Result'])
copyDF

# test train spirit
X = copyDF[['Age', 'Gender', 'TB', 'DB', 'AAP', 'SAA', 'TP', 'ALB', 'A/G', 'S']]
y = copyDF[['Result']]
from sklearn.model_selection import train_test_split
#X_train, x_test, y_train, y_test = train_test_split(df2, test_size = 0.2, random_
state = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_s
tate = 1
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Isolation Forests
from sklearn.ensemble import IsolationForest
clf = IsolationForest(max_samples=100, contamination=0.0
df_drop = df.drop(axis=1, index=1)
df_drop.head()
df_drop_na = df_drop.dropna()
#clf.fit(np.vstack((df_drop_na.iloc[:, 8].values, df_drop_na.iloc[:, 9].values)).T)
clf.fit(X_train, y_train)
#clf_pred = clf.predict(np.vstack((df_drop_na.iloc[:, 8].values, df_drop_na.iloc[:
, 9].values)).T)
clf_pred = clf.predict(X_test)
clf_pred

# decision trees
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred
accuracy_score(y_test, y_pred)

# logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=1, max_iter = 10000).fit(X_train, y_train
.to_numpy().reshape(463, ))
y_pred = clf.predict(X_test)
y_pred
accuracy_score(y_test, y_pred)
y_train.to_numpy().reshape(463, 1)

#random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train.to_numpy().reshape(463, ))
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# SVM
from sklearn import svm
model = svm.SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test) accuracy_score(y_test, y_pred)