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
    
def median(df,var):
    df[var] = df[var].fillna(df[var].median())
    return df
    
def cmean(df,var,features):
    df[var] = df.groupby(features)[var].transform(lambda x: x.fillna(x.mean()))
    return df

def zero(df):
    df[var] = df[var].fillna(0)
    return df

import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words('english')
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
