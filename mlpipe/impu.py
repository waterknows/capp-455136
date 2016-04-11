import pandas as pd
import numpy as np
import pylab as pl
from sklearn.neighbors import KNeighborsRegressor

def KNN(df,var,x,n):
#var=variable being imputed; x=number of most correlated variables used to predict KNN, n=neighbors
	var_imputer = KNeighborsRegressor(n_neighbors=n)
	df_full = df[df[var].isnull()==False]
	df_null = df[df[var].isnull()==True]
	cols=list(df.corr().loc[:,var].abs().sort_values(ascending=False)[1:x+1].index)
	var_imputer.fit(df_full[cols], df_full[var])
	impute = var_imputer.predict(df_null[cols])
	df_null[var] = impute
	df = df_full.append(df_null)
	return df

def mean(df,var):
	df[var] = df[var].fillna(df[var].mean())
	return df

def cmean(df,var,var1):
	df[var] = df.groupby(var1)[var].transform(lambda x: x.fillna(x.mean()))
	return df

def zero(df):
	df[var] = df[var].fillna(0)
	return df
