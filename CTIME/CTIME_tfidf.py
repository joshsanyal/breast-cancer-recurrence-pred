import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from Preprocessing import process
from xgboost import XGBClassifier
from Preprocessing import parse_impression

### PROCESSING (X = reports, Y = y)
CTIMEreports = pd.read_csv("data/CTIME/CTIMEFinalLabel.csv")
processed = []
for report in CTIMEreports['CTReport']:
    report = parse_impression(report)
    rep = ' '.join(process(report))
    processed.append(rep)
CTIMEreports['PROCESSED'] = (processed)


### BUILD MODEL (can change fit & predict functions to train on 'CTReport' (unprocessed), 'PROCESSED', 'FINDINGS' (processed), IMPRESSION' (processed)

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=200)),('tfidf', TfidfTransformer()),
                 ('clf', XGBClassifier(max_depth = 15, eta = 0.1, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2)),])
train, test = train_test_split(CTIMEreports, test_size=0.20, random_state=235) #random_state fixes randomness across trials
text_clf = text_clf.fit(train['PROCESSED'], train['IC_Mass_Effect'])
pred = text_clf.predict(test['PROCESSED'])

### GET RESULTS
print("PRECISION")
print(precision_score(test['IC_Mass_Effect'],pred))
print()
print("RECALL")
print(recall_score(test['IC_Mass_Effect'],pred))
print()
print("F1")
print(f1_score(test['IC_Mass_Effect'],pred))
print()
print("Accuracy")
print(accuracy_score(test['IC_Mass_Effect'],pred))
print()


### SAVE MODEL WEIGHTS + PREDICTIONS
'''
list_of_tuples = list(zip(text_clf['vect'].vocabulary_, text_clf['clf'].coef_[0]))
EN = pd.DataFrame(list_of_tuples, columns = ['Word', 'Elastic Net Coef'])
EN.to_csv("data/CTIME/CTIME_Elastic_Net.csv")
'''
