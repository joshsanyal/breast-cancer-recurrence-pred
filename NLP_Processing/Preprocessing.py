import nltk
from nltk.corpus import stopwords
import re
import string
from SentTokenizer import segment
from nltk.stem import WordNetLemmatizer
from dateutil.parser import parse
from dateparser.search import search_dates
from dateutil.relativedelta import relativedelta
import spacy
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

def text2embed():
    processedNotes = pd.read_csv("data/OncoShare/ProcessedNotesLabelled2.csv", lineterminator='\n')
    reports = processedNotes["PROCESSED"]
    tfidf = TfidfVectorizer(sublinear_tf=True)
    features = tfidf.fit_transform(processedNotes["PROCESSED"])
    feature_names = tfidf.get_feature_names()
    df = pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())

    preTrainedPath = "oncoshare_w2v_epoch2.bin"
    wv = KeyedVectors.load_word2vec_format(preTrainedPath, binary=True)
    feats = []

    for i in range(len(reports)):
        if (i%1000==0): print(i)
        report = reports[i].split()
        avgFeat = np.zeros(300)
        counter = 1
        for word in report:
            try: # if in vocab
                vector = np.multiply(wv[word],df[word][i])
                avgFeat = np.add(avgFeat, vector)
                counter += 1
            except: # if out of vocab
                '''
                '''
        feats.append(avgFeat/counter)
    return feats


def parse_impression(full_report):
    '''
    Return the impression given the full text of the report
    or empty string if impression fails to parse
    Args:
        full_report : string representing the full report text
    Returns:
        string denoting the impression parsed from the full report.
        all words are converted to lower case
    '''
    impression_words = []
    #all_words = re.findall(r"[w']+", full_report)
    all_lines = full_report.split('.')
    start = False
    for index in range(len(all_lines)):
        line = all_lines[index].lower().strip()
        if len(line) == 0:
            continue
        if ('impression:' in line or 'findings:' in line or '1.' in line):
            start = True
        if start and ('report release date' in line     or 'i have reviewed' in line     or 'electronically reviewed by' in line    or 'electronically signed by' in line    or 'attending md' in line      or 'electronic signature by:' in line):
            break
        if start or 'mass effect' in line or 'midline shift' in line or 'hemorr' in line or 'hematoma' in line or 'hernia' in line or 'sah' in line:
            impression_words.append(line +'.')
    # Check if parsing failed
    if len([word for line in impression_words for word in line.split()]) < 2:
        return ''
    else:
        return '\n'.join(impression_words)


### PROCESSING BEFORE TRAINING SEG ALG
def prelimProcess(text):
    # STANDARDIZE WHITESPACES
    spaces = "                         "
    for i in range(22):
        text = text.replace(spaces[i:],"\n")
    newlines = "\n\n\n\n\n\n\n\n\n\n"
    for i in range(10):
        text = text.replace(newlines[i:],"\n")

    # REMOVE DATES
    start = text.find("[**")
    while start != -1:
        end = text.find("**]", start)
        if((text[start+3:start+7]).isnumeric()): # does it begin w/ a year?
            text = text.replace(text[start:end+3],"date") # remove dates
        else:
            text = text.replace(text[start:end+3],"propernoun") # replace w/ propernoun
        start = text.find("[**")
    return text


### Returns difference between note date and event date in note
def diffDates(d2, d1):
    diffDays = abs((d2 - d1).days)
    if diffDays <= 30: return (str(diffDays) + " day ago")
    elif (diffDays <= 364): return (str(relativedelta(d2,d1).months) + " month ago")
    return (str(relativedelta(d2,d1).years) + " year ago")

### Named Entity Recognition
def NER(text):
    sp = spacy.load('en_core_web_sm')
    doc = sp(text)
    for X in doc.ents:
        text = text.replace(" " + X.text + " ", " " + X.label_ + " ")
    return text

### PROCESSING AFTER TRAINING SEG ALG
def process(text):
    data = open("data/CLEVER.txt","r")
    line = []
    for i in range(1368):
        line.append(data.readline()[:-1])
    data.close()
    for i in reversed(range(1368)):
        term = line[i].split(sep = "|")
        text = text.replace(" " + term[1] + " ", " " + term[2] + " ")
        text = text.replace(" " + term[1] + ".", " " + term[2] + ".")
        text = text.replace(" " + term[1] + ",", " " + term[2] + ",")

    text = segment(prelimProcess(text))
    exclude_list = stopwords.words('english')

    lemmatizer = WordNetLemmatizer()

    popList = []
    counter = 0

    for i in range(len(text)):
        numNumbers = 0
        text[i] = text[i].replace("\n", " ")
        text[i] = re.sub('['+ string.punctuation + ']', ' ', text[i]) #remove punctuation
        tagged_sent = pos_tag(text[i].split())
        ppns = [word for word,pos in tagged_sent if pos == 'NNP']
        for ppn in ppns:
            text[i] = text[i].replace(ppn, "")

        text[i] = text[i].lower()

        ''' REPLACE DATES WITH XX MONTHS AGO, etc.
        dates = search_dates(text[i])
        if dates is not None:
            for date in dates:
                text[i] = text[i].replace(date[0], diffDates(noteDateTime,date[1]))
        '''

        words = text[i].split(' ')
        sent = []
        for j in range(len(words)):
            if (words[j] not in exclude_list and len(words[j]) > 2):
                word = lemmatizer.lemmatize(words[j]) # lemmatization
                if (word.isnumeric()):
                    numNumbers += len(int2word(int(word)).split())
                    #for w in int2word(int(word)).split(): # convert num to words
                        #sent.append(w)
                elif (word in exclude_list or len(word) <= 2): pass # remove stopwords + (1,2-letter words)
                else: sent.append(word)
        text[i] = " ".join(sent)

        if (len(text[i]) <= 2 or numNumbers > len(text[i])/2):
            popList.append(i-counter)
            counter += 1

    for index in popList:
        text.pop(index) # remove all sentence with <= 3 words

    return text




def removeSections(text):
    return text[text.find('Service:'):text.rfind('Dictated')]

def normalHeaders(text):
    index = text.find(":")
    while index != -1:
        spaceIndex = index - 1
        char = text[spaceIndex]
        while char != " ":
            spaceIndex -= 1
            char = text[spaceIndex]
        word = text[spaceIndex+1:index+1]
        if not word.islower():
            text.replace(word, "")
        index = text.find(":", index+1)
    return text

######### Replace proper nouns
def POStagging(text):
    text = nltk.word_tokenize(text)
    result = nltk.pos_tag(text)
    text = ""
    for a in result:
        if (a[0] == "propernoun"): text = text + "propernoun|NNP "
        elif (a[0] == "date"): text = text + "date|NNP "
        elif (a[1] != "NNP"): text = text + a[0] + "|" + a[1] + " "
        else: text = text + "propernoun|NNP ";
    return text


def int2word(n):
    """
    convert an integer number n into a string of english words
    """
    # break the number into groups of 3 digits using slicing
    # each group representing hundred, thousand, million, billion, ...
    n3 = []
    r1 = ""
    # create numeric string
    ns = str(n)
    for k in range(3, 33, 3):
        r = ns[-k:]
        q = len(ns) - k
        # break if end of ns has been reached
        if q < -2:
            break
        else:
            if  q >= 0:
                n3.append(int(r[:3]))
            elif q >= -1:
                n3.append(int(r[:2]))
            elif q >= -2:
                n3.append(int(r[:1]))
        r1 = r

    #print n3  # test

    # break each group of 3 digits into
    # ones, tens/twenties, hundreds
    # and form a string
    nw = ""
    for i, x in enumerate(n3):
        b1 = x % 10
        b2 = (x % 100)//10
        b3 = (x % 1000)//100
        #print b1, b2, b3  # test
        if x == 0:
            continue  # skip
        else:
            t = thousands[i]
        if b2 == 0:
            nw = ones[b1] + t + nw
        elif b2 == 1:
            nw = tens[b1] + t + nw
        elif b2 > 1:
            nw = twenties[b2] + ones[b1] + t + nw
        if b3 > 0:
            nw = ones[b3] + "hundred " + nw
    return nw


### globals
ones = ["", "one ","two ","three ","four ", "five ", "six ","seven ","eight ","nine "]
tens = ["ten ","eleven ","twelve ","thirteen ", "fourteen ", "fifteen ","sixteen ","seventeen ","eighteen ","nineteen "]
twenties = ["","","twenty ","thirty ","forty ", "fifty ","sixty ","seventy ","eighty ","ninety "]
thousands = ["","thousand ","million ", "billion ", "trillion ", "quadrillion ", "quintillion ", "sextillion ", "septillion ","octillion ",
    "nonillion ", "decillion ", "undecillion ", "duodecillion ", "tredecillion ", "quattuordecillion ", "quindecillion", "sexdecillion ",
    "septendecillion ", "octodecillion ", "novemdecillion ", "vigintillion "]
