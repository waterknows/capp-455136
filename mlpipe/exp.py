import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pylab as pl
import numpy as np

def dfsum(df):
	stats=df.describe(include='all').transpose()
	stats['missing']=1-stats['count']/len(df.index)
	stats['median']=df.median()
	stats['mode']=df.mode().transpose()[0]
	table=stats[['mean','median','mode','std','missing']]
	return table

def dfcorr(df):
	corrmatrix=df.corr(method='pearson')
	return corrmatrix

def dfhist(df):
    hist=df.hist(color='b', alpha=0.5,bins=50, figsize=[20,10])
    return hist

def dfcorrp(df):
	heat=plt.colorbar(plt.matshow(df.corr(method='pearson')))
	return heat

def dfforest(df,var):
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

def cap(x, cap):
    if x > cap:
        return cap
    else:
        return x

def disc(df,var,x):
	df[var+'bin'] = pd.cut(df[var], bins=x, labels=False)
	return df[str(var)+'bin']

