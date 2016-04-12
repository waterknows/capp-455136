import pandas as pd
import numpy as np
import pylab as pl
from sklearn.neighbors import KNeighborsRegressor

def KNN(df,var,features,k):
    var_imputer = KNeighborsRegressor(n_neighbors=k)
    df_full = df[df[var].isnull()==False]
    df_null = df[df[var].isnull()==True]
    var_imputer.fit(df_full[features], df_full[var])
    impute = var_imputer.predict(df_null[features])
    df_null[var] = impute
    df = df_full.append(df_null)
    return df

def mean(df,var):
    df[var] = df[var].fillna(df[var].mean())
    return df

def cmean(df,var,features):
    df[var] = df.groupby(features)[var].transform(lambda x: x.fillna(x.mean()))
    return df

def zero(df):
    df[var] = df[var].fillna(0)
    return df
