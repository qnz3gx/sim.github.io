import pandas as pd
import numpy as np
import plotly.express as px

def import_csv_with_pandas(file_path,r):
    data = pd.read_csv(file_path, skiprows=r)
    return data

file_path = '/Users/scarlettimorse/PycharmProjects/PythonProject/Website/Available_Polarized_Data.csv'
APD_df = import_csv_with_pandas(file_path,10)

experiment_mapping = {
    1: 'SLAC E142',
    2: 'SLAC E154',
    3: 'Zheng',
    4: 'Kramer',
    5: 'Flay',
    6: 'Solvignon',
    7: 'E97110',
    8: 'E94010',
}

Q2_groups = {
    q2: df for q2, df in APD_df.groupby('Q2')
}

unique_q2=sorted(APD_df['Q2'].unique())
unique_q2.remove(-1000.0)

N = 5
fig = px.scatter(Q2_groups[unique_q2[N]][(Q2_groups[unique_q2[N]].W > -999) & (Q2_groups[unique_q2[N]]['G1.mes'] > -999)], x='W', y='G1.mes', title='G1 vs W')
fig.write_html("plotly_figure.html")
print(unique_q2[N])