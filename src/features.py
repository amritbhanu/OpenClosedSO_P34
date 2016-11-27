from __future__ import print_function, division

__author__ = 'amrit'

import sys
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import lda

sys.dont_write_bytecode = True

def vectorize(records, vocab=None):
    if vocab is None:
        vectorizer = CountVectorizer()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
    return vectorizer.fit_transform(records), vectorizer

"tf-idf"
def tfidf(corpus):
    vectorizer = TfidfTransformer()
    return vectorizer.fit_transform(corpus)


"tf-idf"
def tf_idf(corpus):
    word={}
    doc={}
    docs=0
    for row_c in corpus:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    tfidf={}
    words=sum(word.values())
    for key in doc.keys():
        tfidf[key]=word[key]/words*np.log(docs/doc[key])
    return tfidf

"tf-idf_incremental"
def tf_idf_inc(row,word,doc,docs):
    docs+=1
    for key in row.keys():
        try:
            word[key]+=row[key]
        except:
            word[key]=row[key]
        try:
            doc[key]+=1
        except:
            doc[key]=1

    return word,doc,docs

"L2 normalization_row"
def l2normalize(mat):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,2)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat

"hashing trick"
def hash(mat,n_features=100, non_negative=True):
    if type(mat[0])==type('str') or type(mat[0])==type(u'unicode'):
        hasher = FeatureHasher(n_features=n_features, input_type='string', non_negative=non_negative)
    else:
        hasher = FeatureHasher(n_features=n_features, non_negative=non_negative)
    X = hasher.transform(mat)
    return X

"make feature matrix"
def make_feature(corpus, sel="tfidf",norm="l2row",n_features=10000):
    dict = []
    for i in corpus:
        dict.append(Counter(i.lower().split()))

    if sel=="hash":
        matt=hash(dict,n_features=n_features,non_negative=True)
        matt=l2normalize(matt).toarray()

    elif sel=="lda":
        corpus, _ = vectorize(corpus)
        lda1=lda.LDA(n_topics=500, alpha=0.1, eta=0.01, n_iter=1000)
        lda1.fit_transform(corpus)
        matt = lda1.doc_topic_

    else:
        score={}
        score=tf_idf(dict)
        keys=np.array(score.keys())[np.argsort(score.values())][-n_features:]
        data=[]
        r=[]
        col=[]
        num=len(dict)
        for i,row in enumerate(dict):
            tmp=0
            for key in keys:
                if key in row.keys():
                    data.append(row[key])
                    r.append(i)
                    col.append(tmp)
                tmp=tmp+1
        dict=[]
        matt=csr_matrix((data, (r, col)), shape=(num, n_features))
        data=[]
        r=[]
        col=[]
        if norm=="l2row":
            matt=l2normalize(matt).toarray()

    return matt

