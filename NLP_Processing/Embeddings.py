import os
from time import time
from gensim.models import FastText

### Get sentences for training
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('processed')

'''
### Get pre-trained word vectors
preTrainedPath = "mimic-pubmed_2.bin"
t = time()
pubmed_wv = KeyedVectors.load_word2vec_format(preTrainedPath, binary=True)
print('Time to read the model: {} mins'.format(round((time() - t) / 60, 2)))
print('----------------------------')
print("ORIGINAL:")
print(pubmed_wv.most_similar(positive=['treatment']))
print(pubmed_wv.most_similar(positive=['female']))
print(pubmed_wv.most_similar(positive=['history']))
print(pubmed_wv.most_similar(positive=['disease']))
print(pubmed_wv.most_similar(positive=['brain']))
print('----------------------------')
'''

### Create word2vec model w/ merged vocab
t = time()
new_wv = FastText(size=30, window=5, min_count=1, workers=3, sg=0, hs=1, negative = 10, sample=0.001, alpha=0.1)
new_wv.build_vocab(sentences)
'''
total_examples = new_wv.corpus_count
new_wv.build_vocab([list(pubmed_wv.vocab.keys())], update=True)
new_wv.intersect_word2vec_format(preTrainedPath, binary=True, lockf=1.0)
'''

### Train for 2 epochs
new_wv.train(sentences, epochs=2) # , total_examples=total_examples
print('Time to train the model 2 epochs: {} mins'.format(round((time() - t) / 60, 2)))
print('----------------------------')
print(new_wv.most_similar(positive=['treatment']))
print(new_wv.most_similar(positive=['female']))
print(new_wv.most_similar(positive=['history']))
print(new_wv.most_similar(positive=['disease']))
print(new_wv.most_similar(positive=['brain']))
new_wv.save_word2vec_format('mimic-pubmed_2.bin', binary=True)
print('----------------------------')


# Train for 10 epochs
new_wv.train(sentences, epochs=8) # , total_examples=total_examples
print('Time to train the model 10 epochs: {} mins'.format(round((time() - t) / 60, 2)))
print('----------------------------')
print(new_wv.most_similar(positive=['treatment']))
print(new_wv.most_similar(positive=['female']))
print(new_wv.most_similar(positive=['history']))
print(new_wv.most_similar(positive=['disease']))
print(new_wv.most_similar(positive=['brain']))
new_wv.save_word2vec_format('mimic-pubmed_10.bin', binary=True)
print('----------------------------')

# Train for 20 epochs
new_wv.train(sentences, epochs=10) # , total_examples=total_examples
print('Time to train the model 20 epochs: {} mins'.format(round((time() - t) / 60, 2)))
print('----------------------------')
print(new_wv.most_similar(positive=['treatment']))
print(new_wv.most_similar(positive=['female']))
print(new_wv.most_similar(positive=['history']))
print(new_wv.most_similar(positive=['disease']))
print(new_wv.most_similar(positive=['brain']))
new_wv.save_word2vec_format('mimic-pubmed_20.bin', binary=True)
print('----------------------------')
