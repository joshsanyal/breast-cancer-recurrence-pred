import pandas as pd
from Preprocessing import process

### TEST
'''
MIMICnotes = pd.read_csv("data/MIMIC_data/NOTEEVENTS.csv", nrows=5)
note = MIMICnotes['TEXT']
#print(note[1])
print(process(note[4]))
'''

'''
CTIMEreports = pd.read_excel('data/CTIME/CTIME_reports.xlsx', sheet_name='CTIME_New')
reports = CTIMEreports['report']
'''

CTIMEreports = pd.read_excel('data/CTIME/CT_MRI_rpt_DE_ID.xlsx', sheet_name=0)
reports = CTIMEreports['report']

### PRE-PROCESSING
'''
for i in range(768,2000):
    print(i)
    if (i == 0):
        MIMICnotes = pd.read_csv("data/MIMIC_data/NOTEEVENTS.csv", nrows=1000)
    else:
        MIMICnotes = pd.read_csv("data/MIMIC_data/NOTEEVENTS.csv", nrows=1000, skiprows=[2,i*1000])
    MIMICnotes['PROCESSED'] = MIMICnotes['TEXT'].apply(process)
    MIMICnotes['PROCESSED'].to_csv("processed/MIMIC_" + str(i), index=False)
'''

'''
CTIMEreports['PROCESSED'] = CTIMEreports['report'].apply(process)
CTIMEreports['PROCESSED'].to_csv("data/CTIME/unnanotated_processed.csv", index=False)
'''
CTIMEreports['PROCESSED'] = CTIMEreports['report'].apply(process)
CTIMEreports['PROCESSED'].to_csv("data/CTIME/processed2", index=False)

