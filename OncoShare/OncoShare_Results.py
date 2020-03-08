import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, auc, roc_curve

def showGraph(patient):
    register_matplotlib_converters()

    pred = np.load("data/OncoShare/yPred.npy")
    testY = np.load("data/OncoShare/yActual.npy")
    testDates = np.load("data/OncoShare/TestNoteDates.npy", allow_pickle=True)

    predGraph = []
    predDate = []
    realGraph = []
    realDate = []
    counter = 0
    date = testDates[patient][counter]
    preds = []
    real = 0
    while counter < 800 and testY[patient][counter][2] == 0:
        if ((testDates[patient][counter] - date).days < 60):
            preds.append(pred[patient][counter][1] / (1-pred[patient][counter][2]))
            if (testY[patient][counter][1] == 1): real = 1
        else:
            sum = 0
            for a in preds:
                sum += a
            predGraph.append(sum/len(preds))
            predDate.append(date)
            realGraph.append(real)
            realDate.append(date)
            date = testDates[patient][counter]
            preds = [pred[patient][counter][1] / (1-pred[patient][counter][2])]
        counter += 1
    plt.figure()
    plt.ylim((-0.05, 1.05))
    plt.plot(predDate, predGraph, 'bo-', label='Predicted Recurrence (Probability)')
    plt.plot(realDate, realGraph, 'ro-', label='Actual Recurrence (Binary)')
    plt.legend()
    plt.gcf().autofmt_xdate()

    plt.xlabel("Date of Visit")
    plt.ylabel("Probability of Breast Cancer Recurrence (1 year)")
    plt.show()

def getPreds(startIndex, endIndex):
    pred = np.load("data/OncoShare/yPred.npy")
    testY = np.load("data/OncoShare/yActual.npy")

    preds = []
    actuals = []
    for i in range(pred.shape[0]):
        for j in range(startIndex, endIndex):
            if (testY[i][j][2] == 0):
                preds.append(pred[i][j][1] / (1 - pred[i][j][2]))
                actuals.append(testY[i][j][1])
    return (preds, actuals)


roc = []
sens = []
spec = []
bins = 20
for i in range(8):
    preds, actuals = getPreds(i*int(800/bins),(i+1)*int(800/bins))
    fpr, tpr, threshold = roc_curve(actuals, preds)
    roc.append(auc(fpr, tpr))

    i = np.arange(len(tpr))
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]

    binPred = []
    for pred in preds:
        if (pred > optimal_threshold): binPred.append(1)
        else: binPred.append(0)
    binPred = np.array(binPred)

    sens.append(recall_score(actuals,binPred))
    tn, fp, fn, tp = confusion_matrix(actuals,binPred).ravel()
    spec.append(tn/(tn+fp))

labels = ['1-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, roc, width, label='ROC AUC', color = "teal")
rects2 = ax.bar(x, sens, width, label='Sensitivity', color = "gray")
rects3 = ax.bar(x + width, spec, width, label='Specificity', color = "darkred")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()




'''
for patient in range(0,50):
    showGraph(patient)
'''

'''
testY = testY.reshape(testY.shape[0]*testY.shape[1],3)
pred = pred.reshape(pred.shape[0]*pred.shape[1],3)

deleteIndex = []
for i in range(testY.shape[0]):
    if (testY[i][2] == 1):
        deleteIndex.append(i)

testY = np.delete(testY, deleteIndex, 0)
pred = np.delete(pred, deleteIndex, 0)


fpr, tpr, _ = metrics.roc_curve(testY[:, 1], pred[:, 1])
roc_auc = metrics.auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''
