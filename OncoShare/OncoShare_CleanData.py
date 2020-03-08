import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from OncoShare_PrepData import str2Date
from Preprocessing import process, text2embed
from xgboost import XGBClassifier
import numpy as np
from Preprocessing import parse_impression
from datetime import datetime, timedelta
from nltk.tag import pos_tag
from sklearn.linear_model import LogisticRegression
from numpy import zeros

### Initial Compilation
'''
y = pd.read_csv("data/OncoShare/Patient_RECUR_quater_v4.csv")
#NoteReports = pd.read_csv("data/OncoShare/STANFORD_NOTE_DATA_TABLE.csv", nrows = 10)
#PathReports = pd.read_csv("data/OncoShare/STANFORD_PATHOLOGY_DATA_TABLE.csv", nrows = 8)

column_names = ["ANON_ID", "NOTE_DATE", "NOTE_TYPE", "NOTE"]
combined_df = pd.DataFrame(columns = column_names)
index = 0
ids =  [False for i in range(100000)]

for i in range(0,len(y)):
    ids[int(y["ANON_ID"][i])] = True

for i in range(0,274):
    print(i)
    notes = []
    if (i == 0):
        notes = pd.read_csv("data/OncoShare/STANFORD_NOTE_DATA_TABLE.csv", nrows=10000, encoding="ISO-8859-1")
    else:
        notes = pd.read_csv("data/OncoShare/STANFORD_NOTE_DATA_TABLE.csv", nrows=10000, skiprows=[i for i in range(1,i*10000)], encoding="ISO-8859-1")
    for j in range(0,len(notes)):
        #label = y.loc[y["ANON_ID"] == notes["ANON_ID"][j]]
        if ids[notes["ANON_ID"][j]]:
            try:
                if "/" in notes["NOTE_DATE"][j] or "-" in notes["NOTE_DATE"][j]:
                    combined_df.loc[index] = [notes["ANON_ID"][j], notes["NOTE_DATE"][j], notes["NOTE_TYPE"][j], notes["NOTE"][j]]
                    index += 1
            except:
                pass
    if (i % 10 == 9):
        combined_df.to_csv("data/OncoShare/NoteDataset" + str(i) + ".csv", index=False)
        combined_df = pd.DataFrame(columns = column_names)
        index = 0
'''

### Merging
'''
column_names = ["ANON_ID", "NOTE_DATE", "NOTE_TYPE", "NOTE"]
combined_df = pd.DataFrame(columns = column_names)

for i in range(0,11):
    print(i)
    if i == 0: ds = pd.read_csv("data/OncoShare/NoteDataset9.csv")
    else: ds = pd.read_csv("data/OncoShare/NoteDataset" + str(i) + "9.csv")
    combined_df = combined_df.append(ds)

combined_df.to_csv("data/OncoShare/NoteDataset.csv", index=False)
'''

### Text Processing
'''
notes = pd.read_csv("data/OncoShare/NoteDatasetLabelled.csv")

processed = []
for report in notes['NOTE']:
    if (len(processed) % 1000 == 0): print(len(processed))
    rep = ' '.join(process(report))
    processed.append(rep)

notes["PROCESSED"] = processed
notes.to_csv("data/OncoShare/ProcessedNotesLabelled.csv", index=False)
'''

### Eliminate notes which aren't strings
'''
processedNotes = pd.read_csv("data/OncoShare/ProcessedNotesLabelled.csv", lineterminator='\n')
column_names = ["ANON_ID", "NOTE_DATE", "NOTE_TYPE", "NOTE", "LABEL", "PROCESSED"]
df = pd.DataFrame(columns = column_names)
index = 0
for i in range(len(processedNotes)):
    if (i%1000 == 0): print(i)
    if isinstance(processedNotes["PROCESSED"][i], str):
        df.loc[index] = [processedNotes["ANON_ID"][i], processedNotes["NOTE_DATE"][i], processedNotes["NOTE_TYPE"][i], processedNotes["NOTE"][i], processedNotes["LABEL"][i], processedNotes["PROCESSED"][i]]
        index += 1
df.to_csv("data/OncoShare/ProcessedNotesLabelled2.csv", index=False)
'''

### Labelling (x months in the future)
def label(future):
    x = pd.read_csv("data/OncoShare/ProcessedNotesLabelled2.csv", lineterminator='\n')
    y = pd.read_csv("data/OncoShare/Patient_RECUR_quater_v4.csv")

    index = [-1 for i in range(100000)]
    x["LABEL"+str(future)+"MONTHS"] = [1 for i in range(len(x))]

    for labelIndex in range(len(y)):
        if index[y["ANON_ID"][labelIndex]] == -1:
            index[y["ANON_ID"][labelIndex]] = labelIndex

    for noteIndex in range(len(x)):
        if (noteIndex % 1000 == 0): print(noteIndex)
        noteDate = str2Date(x["NOTE_DATE"][noteIndex])
        for labelIndex in range(index[x["ANON_ID"][noteIndex]], len(y)):
            if (y["ANON_ID"][labelIndex] == x["ANON_ID"][noteIndex]):
                labelDate = datetime.strptime(y["DATE"][labelIndex], '%Y-%m-%d')
                diff = labelDate - noteDate
                if (diff.days > 30*future):
                    if ("Definite recurrence" not in y["RECUR"][labelIndex]): #no recurrence + suggestive = 0
                        x["LABEL"+str(future)+"MONTHS"][noteIndex] = 0
                    break
    x.to_csv("data/OncoShare/ProcessedNotesLabelled2.csv", index=False)
label(9)

### TF-IDF * word2vec (average)
'''
vec = np.array(text2embed())
np.save("data/OncoShare/w2vDataset", vec)
'''


