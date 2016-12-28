import pandas as pd
import numpy as np
from numpy import *
import operator
import collections
import itertools
#upload train and test sets
pe_Dataset1=pd.read_csv('C:/Users/chaithu/Desktop/trainset.csv')
pe_test=pd.read_csv('C:/Users/chaithu/Desktop/testset.csv')
a=list(pe_Dataset1.columns.values)
b=list(pe_test.columns.values)
c=len(a)
d=len(b)
print(c)
print(d)
g=list(pe_Dataset1[a[c-1]])
del pe_Dataset1[a[c-1]]   
h=list(pe_test[b[d-1]])
del pe_test[b[d-1]]
o= pd.DataFrame(g)
y_train=o
p= pd.DataFrame(h)
y_test=p
x_train=pe_Dataset1
x_test=pe_test
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(x_train, y_train)
Test = clf_tree.predict(x_test)
confusion = confusion_matrix(y_test, Test)
print(confusion)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(x_train, y_train)
y_pred_class = knn.predict(x_test)
confusion_knn = confusion_matrix(y_test, y_pred_class)
print(confusion_knn)
from sklearn.ensemble import RandomForestClassifier
rand_class = RandomForestClassifier(n_estimators=100)
rand_class.fit(x_train, y_train)
y_pred_class_rand = rand_class.predict(x_test)
confusion_rand = confusion_matrix(y_test, y_pred_class_rand)
print(confusion_rand)
