import pandas as pd
import numpy as np

apd = '/Users/scarlettimorse/PycharmProjects/sim.github.io/Available_Polarized_Data.csv'
flay = '/Users/scarlettimorse/PycharmProjects/sim.github.io/Flay.csv'
def import_csv(file_path, lines):
    data = pd.read_csv(file_path, skiprows=lines)
    return data

APD_df = import_csv(apd, 10)
Flay_df = import_csv(flay,0)
j=0

APD_df = APD_df.copy()
for i in range(len(APD_df)):
    if APD_df['Experiment'].iloc[i] == 5:
        APD_df['G1.mes'].iloc[i] = Flay_df['G1.mes'].iloc[j]
        APD_df['G1.mes.err'].iloc[i] = Flay_df['G1.mes.err'].iloc[j]
        j = j+1

APD_df.to_csv("threeHedata.csv")