import pandas as pd
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
from Preprocessing import process
import os
from gensim.models import Word2Vec, KeyedVectors, FastText
import re
import string
from time import time

'''
t = time()

class callback(CallbackAny2Vec):
    ### Callback to print loss after each epoch.

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        print('Time to train this epoch: {} mins'.format(round((time() - t) / 60, 2)))
        model.wv.save_word2vec_format('oncoshare_w2v_epoch' + str(self.epoch + 1) + '.bin', binary=True)
        self.epoch += 1


### Get sentences for training
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                for sentence in line.split("', '"):
                    sentence = re.sub('['+ string.punctuation + ']', '', sentence)
                    yield sentence.split()


sentences = MySentences('data/OncoShare/processed')
print('BUILDING MODEL')
new_wv = Word2Vec(sentences, size=300, window=30, min_count=50, sg=1, compute_loss=True, callbacks=[callback()], iter = 10)
print('Time to build the model (20 epochs): {} mins'.format(round((time() - t) / 60, 2)))
new_wv.wv.save_word2vec_format('oncoshare_w2v_final.bin', binary=True)
'''

'''
note_reports = pd.read_csv("data/OncoShare/STANFORD_NOTE_DATA_TABLE.csv", nrows=10)
annotations = pd.read_csv("data/OncoShare/Patient_RECUR_quater_v4.csv", nrows=10)

notes = note_reports["NOTE"]
print(notes[1])
for i in range(len(notes)):
    notes[i] = process(notes[i])
print(notes[1])
'''

'''
for i in range(98,274):
    print(i)
    if (i == 0):
        notes = pd.read_csv("data/OncoShare/STANFORD_NOTE_DATA_TABLE.csv", nrows=10000, encoding="ISO-8859-1")
    else:
        notes = pd.read_csv("data/OncoShare/STANFORD_NOTE_DATA_TABLE.csv", nrows=10000, skiprows=[i for i in range(1,i*10000)], encoding="ISO-8859-1")
    notes['PROCESSED'] = notes['NOTE'].apply(process)
    notes['PROCESSED'].to_csv("data/OncoShare/processed/TXT_" + str(i), index=False)
'''


preTrainedPath = "oncoshare_w2v_epoch2.bin"
t = time()
onco_wv = KeyedVectors.load_word2vec_format(preTrainedPath, binary=True)
print('Time to read the model: {} mins'.format(round((time() - t) / 60, 2)))
print('----------------------------')
print(onco_wv.most_similar(positive=['world']))
print(onco_wv.most_similar(positive=['p53']))
print(onco_wv.most_similar(positive=['mitosis']))
print(onco_wv.most_similar(positive=['triple']))
print('----------------------------')
