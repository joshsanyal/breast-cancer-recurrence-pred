from datetime import datetime
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def display_closestwords_tsnescatterplot(words):
    model = KeyedVectors.load_word2vec_format("oncoshare_w2v_epoch2.bin", binary=True)

    arr = np.empty((0,300), dtype='f')
    word_labels = []

    for word in words:
        word_labels.append(word)
        # get close words
        close_words = model.similar_by_word(word)

        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([model[word]]), axis=0)
        for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot

    colors = ["darkcyan", "forestgreen", "mediumvioletred", "coral", "sienna"]
    for i in range(len(words)):
        plt.scatter(x_coords[i*11:(i+1)*11], y_coords[i*11:(i+1)*11], color = colors[i], s = 50, label = words[i] + "-related terms")

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, color="black", xy=(x, y), xytext=(2, 2), size = 14, textcoords='offset points')
    #plt.legend()
    plt.xlim(x_coords.min()-10, x_coords.max()+10)
    plt.ylim(y_coords.min()-10, y_coords.max()+10)
    plt.show()

display_closestwords_tsnescatterplot(["mastectomy", "stage", "biopsy", "gene", "smoke"])


def display_documents_tsnescatterplot():
    vec = np.load("data/OncoShare/w2vDataset.npy")
    processedNotes = pd.read_csv("data/OncoShare/ProcessedNotesLabelled2.csv", lineterminator='\n')

    zeroIndices = []
    oneIndices = []
    counter = 0
    while len(oneIndices) != 20:
        if processedNotes["LABEL1MONTHS"][counter] == 0 and len(zeroIndices) < 20:
            zeroIndices.append(counter)
        elif processedNotes["LABEL1MONTHS"][counter] == 1:
            oneIndices.append(counter)
        counter += 1


    zeroVec = np.array(vec[zeroIndices])
    oneVec = np.array(vec[oneIndices])

    vecs = np.concatenate((zeroVec, oneVec))

    #pca = PCA(n_components=50)
    #pca.fit(vec)
    #vecs = pca.transform(vecs)
    tsne = TSNE(n_components=2, random_state=20)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vecs)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    plt.rcParams.update({'font.size': 16})

    plt.scatter(x_coords[0:20], y_coords[0:20], color = "blue", label = "No recurrence (3 months)")
    plt.scatter(x_coords[20:40], y_coords[20:40], color = "red", label = "Recurrence (3 months)")
    plt.legend()
    plt.show()


#display_documents_tsnescatterplot()
