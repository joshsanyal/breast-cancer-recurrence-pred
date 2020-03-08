import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Input, Flatten, BatchNormalization, Dropout, Bidirectional
from numpy.random import seed
from keras.optimizers import Adam
seed(1)

def sortNotes():
    ret = []
    processedNotes = pd.read_csv("data/OncoShare/ProcessedNotesLabelled2.csv", lineterminator='\n')
    for i in range(len(processedNotes)):
        noteDate = str2Date(processedNotes["NOTE_DATE"][i])
        epoch = datetime.strptime('01/01/80', '%m/%d/%y')
        diff = noteDate - epoch
        add = (processedNotes["ANON_ID"][i], diff.days, i)
        ret.append(add)
    ret.sort()
    return ret

def labelToCategorical(x):
    if x == 0: return [1, 0, 0]
    elif x == 1: return [0, 1, 0]
    else: return [0, 0, 1]

def str2Date(x):
    if("/" in x):
        return datetime.strptime(x, '%m/%d/%y')
    else:
        return datetime.strptime(x, '%d-%m-%y')

def prepData(padVal, label):
    noteVectors = [] # patient_id, note_id, vector_element
    labels = []
    dates = []
    vec = np.load("data/OncoShare/w2vDataset.npy")
    processedNotes = pd.read_csv("data/OncoShare/ProcessedNotesLabelled2.csv", lineterminator='\n')
    sorted = sortNotes()
    curPatID = 0
    noteVecs = []
    labs = []
    dats = []
    for j in range(len(sorted)):
        if (curPatID == sorted[j][0]): # same patient, add to list
            if(len(noteVecs) != padVal):
                noteVecs.append(vec[sorted[j][2]])
                labs.append(labelToCategorical(processedNotes[label][sorted[j][2]]))
                dats.append(str2Date(processedNotes["NOTE_DATE"][sorted[j][2]]))
        else: # diff patient, append old patient info, reset data, add new info
            while(len(noteVecs) != padVal):
                noteVecs.append(np.zeros(300))
                labs.append(labelToCategorical(2))
                dats.append(datetime.now())
            noteVectors.append(noteVecs)
            labels.append(labs)
            dates.append(dats)
            curPatID = sorted[j][0]
            noteVecs = []
            labs = []
            dats = []
            noteVecs.append(vec[sorted[j][2]])
            labs.append(labelToCategorical(processedNotes[label][sorted[j][2]]))
            dats.append(str2Date(processedNotes["NOTE_DATE"][sorted[j][2]]))
    while(len(noteVecs) != padVal):
        noteVecs.append(np.zeros(300))
        labs.append(labelToCategorical(2))
        dats.append(datetime.now())
    noteVectors.append(noteVecs)
    labels.append(labs)
    dates.append(dats)
    return (noteVectors, labels, dates)


### Prep Data in Tensor Dataset
data = prepData(800, "LABEL")
np.save("data/OncoShare/xData", np.array(data[0]))
np.save("data/OncoShare/yData", np.array(data[1]))
np.save("data/OncoShare/noteDates", np.array(data[2]))

#_, testPatients = train_test_split(np.array(data[3]), test_size=0.25, random_state=3) #random_state fixes randomness across trials
#np.save("data/OncoShare/TestPatientIDs", testPatients)

### Distribution of Visits
'''
plt.rcParams.update({'font.size': 18})

cm = plt.cm.get_cmap('RdYlBu_r')
Y,X = np.histogram(noteCounts, 25)
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]
plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.yscale('Log')
plt.xlabel('Number of Visits')
plt.ylabel('Number of Patients (log scaled)')
plt.title('Distribution of Visits Per Patient')
plt.show()
'''
