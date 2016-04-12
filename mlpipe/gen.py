import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def disc(df,var,bin):
    df[str(var)+'bin'] = pd.cut(df[var], bins=15, labels=False)
    return df

def cap(df, x, cap):
    df[x]= df[x].apply(lambda x: x if x < cap else cap)
    return df  

def dummy(df,var):
    df = pd.get_dummies(df[var])
    return df

def norm(df,var):
    df[var] = StandardScaler().fit_transform(df[var])
    return df

def split(df):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    return train,test