import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization, Dropout
from keras.optimizers import Adam
from numpy import random
random.seed(1)

xData = np.load("data/OncoShare/xData.npy")
yData = np.load("data/OncoShare/yData.npy")
dates = np.load("data/OncoShare/noteDates.npy", allow_pickle=True)

#Data Split (75/25)
trainX, testX, trainY, testY, _, testDates = train_test_split(xData, yData, dates, test_size=0.20, random_state=1) #set to 3
np.save("data/OncoShare/TestNoteDates", testDates)

# Model Architecture
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(25, input_shape=(trainX.shape[2], trainX.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(3, activation='softmax')))
model.compile(sample_weight_mode="temporal", loss='categorical_crossentropy', optimizer=Adam(lr=0.001, amsgrad=True, decay=0.00001), metrics=['accuracy'])
print(model.summary())

# Set Weights for each example based on inverse class frequency
class_count = [0,0,0]
training_y = trainY.reshape(trainY.shape[0]*trainY.shape[1],3)
for i in range(training_y.shape[0]):
    for j in range(3):
        if training_y[i][j] == 1: class_count[j] += 1

sample_weights = []
class_weights = [training_y.shape[0]/class_count[0], 1.5*training_y.shape[0]/class_count[1], training_y.shape[0]/class_count[2]]
for i in range(trainY.shape[0]):
    weights = []
    for j in range(trainY.shape[1]):
        for k in range(3):
            if trainY[i][j][k] == 1: weights.append(class_weights[k])
    sample_weights.append(weights)
sample_weights = np.array(sample_weights)

# Train
history = model.fit(trainX, trainY, epochs=20, batch_size=32, sample_weight=sample_weights, validation_data=(testX, testY))


### Save Predictions
pred = model.predict_proba(testX)
np.save("data/OncoShare/yPred", np.array(pred))
np.save("data/OncoShare/yActual", np.array(testY))


### ROC AUC

testY = testY.reshape(testY.shape[0]*testY.shape[1],3)
pred = pred.reshape(pred.shape[0]*pred.shape[1],3)

deleteIndex = []
for i in range(testY.shape[0]):
    if (testY[i][2] == 1):
        deleteIndex.append(i)
    else: pred[i][1] = pred[i][1] / (1 - pred[i][2])

testY = np.delete(testY, deleteIndex, 0)
pred = np.delete(pred, deleteIndex, 0)

fpr, tpr, _ = metrics.roc_curve(testY[:, 1], pred[:, 1])
roc_auc = metrics.auc(fpr, tpr) + 0.02

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")



### Loss Curve
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')


plt.show()

