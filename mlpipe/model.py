import pandas as pd
import numpy as np
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def KNN(train,test,var,features,k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train[features], train[var])
    predict=clf.predict(test[features])
    prob = clf.predict_proba(test[features])
    return predict,prob

def RF(train,test,var,features):
    clf = RandomForestClassifier()
    clf.fit(train[features], train[var])
    predict=clf.predict(test[features])
    prob = clf.predict_proba(test[features])
    return predict,prob