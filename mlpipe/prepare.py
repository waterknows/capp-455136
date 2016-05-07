import pandas as pd
#1.Read
def csvtodf(filename):
    return pd.read_csv(filename)

#2.Train-Validate
def split(df,n):
    msk = np.random.rand(len(df)) < n
    train = df[msk]
    validate = df[~msk]
    return train,validate
    
#3.Keep


#3.Text Data
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
