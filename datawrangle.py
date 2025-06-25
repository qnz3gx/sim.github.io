import pandas as pd
import numpy as np

# one = pd.read_csv('p1.csv')
# two = pd.read_csv('p2.csv')
# three = pd.read_csv('p3.csv')
# four = pd.read_csv('p4.csv')
# five = pd.read_csv('p5.csv')
# six = pd.read_csv('p6.csv')

# clas_eg1b = pd.concat([one,two,three,four,five,six], ignore_index=True)
# clas_eg1b = clas_eg1b.dropna(axis=1, how='all')
# print(clas_eg1b.head())
# clas_eg1b.to_csv('proton_CLAS_EG1b.csv', index=False)

two = pd.read_csv('deuteron_CLAS_EG1.csv')
two['Experiment'] = 'CLAS_EG1'
three = pd.read_csv('deuteron_CLAS_EG1b.csv')
three['Experiment'] = 'CLAS_EG1b'
four = pd.read_csv('deuteron_COMPASS.csv')
four['Experiment'] = 'COMPASS'
five = pd.read_csv('deuteron_HERMES.csv')
five['Experiment'] = 'HERMES'
six = pd.read_csv('deuteron_SLAC_E143.csv')
six['Experiment'] = 'SLAC_E143'
seven = pd.read_csv('deuteron_SMC.csv')
seven['Experiment'] = 'SMC'
eight = pd.read_csv('deuteron_SLAC_E155.csv')
eight['Experiment'] = 'SLAC_E155'

deuteron = pd.concat([two,three,four,five,six,seven,eight], ignore_index=True)

def err_prop(df, col1, col2, col3):
    error = np.sqrt(df[col1].fillna(0).values ** 2 + df[col2].fillna(0).values ** 2 + df[col3].fillna(0).values **2)
    return error

def err_prop2(df, col1, col2):
    error = np.sqrt(df[col1].fillna(0).values ** 2 + df[col2].fillna(0).values ** 2)
    return error

deuteron['dg1(tot)'] = err_prop(deuteron,'dg1(stat)','dg1(sys)','dg1(model)')
deuteron['dg1/F1(tot)'] = err_prop2(deuteron,'dg1/F1(stat)','dg1/F1(sys)')
deuteron['dA1(tot)'] = err_prop(deuteron,'dA1(stat)','dA1(sys)','dA1(model)')

deuteron = deuteron.round(4)

deuteron['dg1(sys)'] = deuteron['dg1(sys)'].replace(0,'')
deuteron['dg1(stat)'] = deuteron['dg1(stat)'].replace(0,'')
deuteron['dg1(model)'] = deuteron['dg1(model)'].replace(0,'')
deuteron['dg1(tot)'] = deuteron['dg1(tot)'].replace(0, '')
deuteron['dg1/F1(stat)'] = deuteron['dg1/F1(stat)'].replace(0,'')
deuteron['dg1/F1(sys)'] = deuteron['dg1/F1(sys)'].replace(0,'')
deuteron['dg1/F1(tot)'] = deuteron['dg1/F1(tot)'].replace(0, '')
deuteron['dA1(stat)'] = deuteron['dA1(stat)'].replace(0,'')
deuteron['dA1(sys)'] = deuteron['dA1(sys)'].replace(0,'')
deuteron['dA1(model)'] = deuteron['dA1(model)'].replace(0,'')
deuteron['dA1(tot)'] = deuteron['dA1(tot)'].replace(0, '')

deuteron = deuteron[['Q2', 'x', 'W',  'Eb', 'g1', 'dg1(stat)', 'dg1(sys)', 'dg1(model)', 'dg1(tot)', 'g1/F1', 'dg1/F1(stat)', 'dg1/F1(sys)', 'dg1/F1(tot)', 'A1', 'dA1(stat)', 'dA1(sys)', 'dA1(model)', 'dA1(tot)','Experiment']]

deuteron.to_csv('DeuteronData.csv', index=False)