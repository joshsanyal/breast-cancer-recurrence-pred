import nltk.tokenize.punkt
import pickle
import pandas as pd
#from Preprocessing import prelimProcess

# TRAIN TOKENIZER
'''
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

MIMICnotes = pd.read_csv("data/MIMIC_data/NOTEEVENTS.csv", nrows=1000000)
notes = MIMICnotes['TEXT'].apply(prelimProcess)

tokenizer.train(notes)
out = open("tokenizer.pk","wb")
pickle.dump(tokenizer, out)
out.close()
'''


def segment(text):
    input = open("tokenizer.pk","rb")
    tokenizerOut = pickle.load(input)
    input.close()
    segmentedNote = tokenizerOut.tokenize(text)
    return segmentedNote

