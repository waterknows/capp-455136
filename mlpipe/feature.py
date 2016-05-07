import pandas as pd
import numpy as np
import pylab as pl
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize


#1.Missing Values
#1)Drop


#2)Univariate Imputation
def impute_mean(df,var):
    df[var] = df[var].fillna(df[var].mean())
    return df
  
def impute_median(df,var):
    df[var] = df[var].fillna(df[var].median())
    return df

def impute_zero(df,var):
    df[var] = df[var].fillna(0)
    return df

#3)Multivariate Imputation
def impute_cmean(df,var,features):
    df[var] = df.groupby(features)[var].transform(lambda x: x.fillna(x.mean()))
    return df


def impute_classifier(df,var,features,classifier):
    var_imputer = classifier
    df_full = df[df[var].isnull()==False]
    df_null = df[df[var].isnull()==True]
    var_imputer.fit(df_full[features], df_full[var])
    impute = var_imputer.predict(df_null[features])
    df_null[var] = impute
    df = df_full.append(df_null)
    return df


def impute_KNN(df,var,features,k,):
    var_imputer = KNeighborsRegressor(n_neighbors=k)
    df_full = df[df[var].isnull()==False]
    df_null = df[df[var].isnull()==True]
    var_imputer.fit(df_full[features], df_full[var])
    impute = var_imputer.predict(df_null[features])
    df_null[var] = impute
    df = df_full.append(df_null)
    return df
    
#2.Discretize
#1)Binary
def discretize_binary(df, x, cap):
    return df[x].apply(lambda x: x if x < cap else cap)
def discretize_dummy(df,var):
    return pd.get_dummies(df[var])
#2)Equal Bins
def discretize_bin(df,var,bin):
    return pd.cut(df[var], bins=15, labels=False)
#3)Equal Size
#4)Entropy Based

#3.Transform
def transform_scale(df,features):
    for f in features:
        df[f+'_scale']=StandardScaler().fit_transform(df[f])
def transform_normall(df,label):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    df_norm[label]=df[label]
    return df_norm

#4.Aggregate

#5.Interact



