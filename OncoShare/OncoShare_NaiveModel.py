import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from Preprocessing import process, text2embed
from xgboost import XGBClassifier
import numpy as np
from Preprocessing import parse_impression
from datetime import datetime, timedelta
from nltk.tag import pos_tag
from sklearn.linear_model import LogisticRegression
from numpy import zeros

###Naive MODEL
'''
processedNotes = pd.read_csv("data/OncoShare/ProcessedNotesLabelled3.csv", lineterminator='\n')
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=200)),('tfidf', TfidfTransformer()),
                 ('clf', XGBClassifier(max_depth = 15, eta = 0.1, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2)),])
train, test = train_test_split(processedNotes, test_size=0.20, random_state=235) #random_state fixes randomness across trials
text_clf = text_clf.fit(train["PROCESSED"], train["LABEL"])
pred = text_clf.predict(test["PROCESSED"])

print("PRECISION")
print(precision_score(test["LABEL"],pred))
print()
print("RECALL")
print(recall_score(test["LABEL"],pred))
print()
print("F1")
print(f1_score(test["LABEL"],pred))
print()
print("Accuracy")
print(accuracy_score(test["LABEL"],pred))
'''

### Word2Vec Model
'''
vec = np.load("data/OncoShare/w2vDataset.npy")
processedNotes = pd.read_csv("data/OncoShare/ProcessedNotesLabelled2.csv", lineterminator='\n')
trainX, testX, trainY, testY = train_test_split(vec, processedNotes["LABEL"], test_size=0.20, random_state=2) #random_state fixes randomness across trials
clf = XGBClassifier(max_depth = 10, eta = 0.1, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2) # max_depth (1,3,5,7,9) n_est (50,100,150,200,250,300)
clf = clf.fit(trainX, trainY)
pred = clf.predict(testX)

### GET RESULTS
print("PRECISION")
print(precision_score(testY,pred))
print()
print("RECALL")
print(recall_score(testY,pred))
print()
print("F1")
print(f1_score(testY,pred))
print()
print("Accuracy")
print(accuracy_score(testY,pred))
'''
