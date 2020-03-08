import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from Preprocessing import process
from xgboost import XGBClassifier
import numpy
from Preprocessing import parse_impression

### PROCESSING (X = reports, Y = y)
CTIMEreports = pd.read_csv("data/CTIME/CTIMEFinalLabel.csv")
AllReports = pd.read_csv("data/CTIME/CTIMEDataset.csv")
processed = []
for report in CTIMEreports['CTReport']:
    report = parse_impression(report)
    rep = ' '.join(process(report))
    processed.append(rep)
CTIMEreports['PROCESSED'] = (processed)

processed = []
for report in AllReports['CTReport']:
    report = parse_impression(report)
    rep = ' '.join(process(report))
    processed.append(rep)
AllReports['PROCESSED'] = processed


### BUILD MODEL (can change fit & predict functions to train on 'CTReport' (unprocessed), 'PROCESSED', 'FINDINGS' (processed), IMPRESSION' (processed)

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=200)),('tfidf', TfidfTransformer()),
                 ('clf', XGBClassifier(max_depth = 15, eta = 0.1, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2)),])
#train, test = train_test_split(CTIMEreports, test_size=0.2) #random_state fixes randomness across trials
text_clf = text_clf.fit(CTIMEreports['PROCESSED'], CTIMEreports['IC_Mass_Effect'])
predict = text_clf.predict_proba(AllReports['PROCESSED'])
pred = []
for i in range(len(AllReports)):
      if (~numpy.isnan(AllReports["IC_Mass_Effect"][i])):
        pred.append(numpy.NaN)
      else:
          pred.append(predict[i][1])
AllReports["Predicted"] = pred

AllReports.to_csv("data/CTIME/CTIMEDatasetClassified.csv")
