import pandas as pd

newRep = pd.read_excel("data/CTIME/ComparePredictedToLabel.xlsx")

numRight = 0
numTotal = 0
reps = pd.read_csv("data/CTIME/CTIMEDatasetClassified.csv")
for i in range(0,len(newRep)):
    if (newRep['CTimeID'][i] != 658 and newRep['CTimeID'][i] != 1508 and newRep['CTimeID'][i] != 1536 and newRep['CTimeID'][i] != 175 and newRep['CTimeID'][i] != 457 and newRep['CTimeID'][i] != 36):
        numTotal += 1
        for j in range(0,len(reps)):
            if newRep['CTimeID'][i] == reps['CTimeID'][j]:
                if newRep['IC_Mass_Effect'][i] == round(reps['Predicted'][j]):
                    numRight += 1
                    break

print(numRight)
print(numTotal)
