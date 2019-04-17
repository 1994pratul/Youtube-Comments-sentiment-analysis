# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:11:39 2019

@author: Pratul Singh
"""
#importing lib
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd
#read the data
data= pd.read_csv(r'C:\Users\admin\Desktop\Downloads\imdb-review-dataset\imdb_master.csv',encoding='latin-1')

data = data.drop('file', 1)
data=data.drop('type',1)

#clean the data from panads
def clean_text(text):
    text=text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

clean_data=data['review'].apply(clean_text)

#now split the data int test and train data

from sklearn.model_selection import train_test_split
clean_data_train, clean_data_test = train_test_split(clean_data, test_size=0.33, random_state=42)

#remove the stop words from dataset
english_stop_words=stopwords.words('english')

def remove_stop_words(corpus):
    remove_stop_words=[]
    for review in corpus:
       remove_stop_words.append(' '.join([word for word in review.split() if word not in english_stop_words ])) 
    return remove_stop_words

no_stop_words_train = remove_stop_words(clean_data_train)
no_stop_words_test = remove_stop_words(clean_data_test)

#using stemming

def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews_train = get_stemmed_text(no_stop_words_train)
stemmed_reviews_test = get_stemmed_text(no_stop_words_test)

#now make our model to make predictions based on our data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))
ngram_vectorizer.fit(stemmed_reviews_train)
X = ngram_vectorizer.transform(stemmed_reviews_train)
X_test = ngram_vectorizer.transform(stemmed_reviews_test)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    
    svm = LinearSVC(C=c)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s"% (c, accuracy_score(y_val, svm.predict(X_val))))
    
final = LinearSVC(C=0.01)
final.fit(X, target)
print ("Final Accuracy: %s"% accuracy_score(target, final.predict(X_test)))


####now its time to print our results
feature_to_coef = {
    word: coef for word, coef in zip(
        ngram_vectorizer.get_feature_names(), final.coef_[0]
    )
}

for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:30]:
    print (best_positive)
    
print("\n\n")
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:30]:
    print (best_negative)
