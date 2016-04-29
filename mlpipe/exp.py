import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pylab as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def summary(df):
    stats=df.describe(include='all').transpose()
    stats['missing']=1-stats['count']/len(df.index)
    stats['median']=df.median()
    stats['mode']=df.mode().transpose()[0]
    table=stats[['mean','median','mode','std','missing']]
    return table

def corr(df):
    corrmatrix=df.corr(method='pearson')
    return corrmatrix

def hist(df):
    hist=df.hist(color='b', alpha=0.5,bins=50, figsize=[20,10])
    return hist

def corrplot(df):
    heat=plt.colorbar(plt.matshow(df.corr(method='pearson')))
    return heat

def mostcorr(df,var,n):
    cols=list(df.corr().loc[:,var].abs().sort_values(ascending=False)[1:n+1].index)
    return cols

def RFplot(df,var):
    features=np.array(df.ix[:, df.columns != var].describe().keys())
    clf = RandomForestClassifier()
    clf.fit(df[features], df[var])
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")
    return pl.show()

def mostRF(df,var,n):
    features=np.array(df.ix[:, df.columns != var].describe().keys())
    clf = RandomForestClassifier()
    clf.fit(df[features], df[var])
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    best_features = features[sorted_idx][::-1]
    return best_features[:n]
    
def display_scores(vectorizer, tfidf_result,n):
    scores = zip(vectorizer.get_feature_names(),np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores[:n]:
        print ("{0:30} Score: {1}".format(item[0], item[1]))
