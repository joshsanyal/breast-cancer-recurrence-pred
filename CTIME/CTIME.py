import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

from Preprocessing import process
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from numpy import empty
from sklearn.metrics import accuracy_score

# Read CTIME reports, get y (output)
CTIMEreports = pd.read_csv("data/CTIME/CTIMEFinalLabel.csv")
processed = []
impression = []
findings = []
for report in CTIMEreports['CTReport']:
    rep = ' '.join(process(report))
    processed.append(rep[rep.find("findings") : rep.find("summary",rep.find("impression"))])
    findings.append(rep[rep.find("findings") : rep.find("impression",rep.find("findings"))])
    impression.append(rep[rep.find("impression") : rep.find("summary",rep.find("impression"))])
CTIMEreports['PROCESSED'] = processed
CTIMEreports['FINDINGS'] = findings
CTIMEreports['IMPRESSION'] = impression
y = CTIMEreports['IC_Mass_Effect']
reports = CTIMEreports['PROCESSED']

# Get word vectors
preTrainedPath = "mimic-pubmed_20.bin"
wv = KeyedVectors.load_word2vec_format(preTrainedPath, binary=True)


# Create x (input features)
numFeatures = 200
oovIDs = {}
oovWords = []
x = empty([296,numFeatures])

for i in range(len(reports)):
    minFeat = np.zeros(numFeatures)
    maxFeat = np.zeros(numFeatures)
    avgFeat = np.zeros(numFeatures)
    words = word_tokenize(reports[i])
    for word in words:
        try: # if in vocab
            vector = wv[word]
            avgFeat = np.add(avgFeat, vector)
        except: # if out of vocab
            '''
            '''
    x[i] = avgFeat/len(words) #np.append() if multiple types
np.save("xCTIME", x)
np.save("yCTIME", y)


# word2vec
x = np.load("xCTIME.npy")
y = np.load("yCTIME.npy")

prec = []
recall = []
f1 = []
acc = []

for i in range(2):
    text_clf = LogisticRegression(penalty='elasticnet',l1_ratio = 0, solver = 'saga', class_weight='balanced', multi_class='multinomial',max_iter=100)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    text_clf = text_clf.fit(train_x, train_y)
    train_pred = text_clf.predict(train_x)
    pred = text_clf.predict(test_x)

    prec.append(precision_score(test_y,pred))
    recall.append(recall_score(test_y,pred))
    f1.append(f1_score(test_y,pred))
    acc.append(accuracy_score(test_y,pred))

print("PRECISION")
print(sum(prec)/len(prec))
print(statistics.stdev(prec))
print()

print("RECALL")
print(sum(recall)/len(recall))
print(statistics.stdev(recall))
print()

print("F1")
print(sum(f1)/len(f1))
print(statistics.stdev(f1))
print()

print("Accuracy")
print(sum(acc)/len(acc))
print(statistics.stdev(acc))
print()



'''
LogisticRegression, PCA – 0.99, RANDOM OOV VECTORS, STANDARDIZED, LOGISTIC REGRESSION, MIN/MAX
TRAINING DATA RESULTS
0.7076271186440678

TEST DATA RESULTS
0.7

LogisticRegression, PCA – 0.95, RANDOM OOV VECTORS, STANDARDIZED, LOGISTIC REGRESSION, MIN/MAX
TRAINING DATA RESULTS
0.7076271186440678

TEST DATA RESULTS
0.7166666666666667


LogisticRegression, PCA – 0.95, STANDARDIZED, LOGISTIC REGRESSION, AVG
TRAINING DATA RESULTS
0.7076271186440678

TEST DATA RESULTS
0.7166666666666667


SVM, PCA – 0.95, STANDARDIZED, LOGISTIC REGRESSION, AVG
TRAINING DATA RESULTS
0.7076271186440678

TEST DATA RESULTS
0.7166666666666667

Decision Tree, SVM, PCA – 0.95, STANDARDIZED, LOGISTIC REGRESSION, AVG
TRAINING DATA RESULTS
0.7076271186440678

TEST DATA RESULTS
0.7166666666666667
'''

