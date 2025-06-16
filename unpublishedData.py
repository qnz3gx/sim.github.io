import pandas as pd
import numpy as np
import plotly.express as px

# E94010_df = pd.read_csv('allData.csv')

# for i in range(len(E94010_df)):
#     if E94010_df['Experiment'].iloc[i] == 'E94010':
#         E94010_df['Q2'].iloc[i] = ''
#         E94010_df['X'].iloc[i] = ''
#         E94010_df['W'].iloc[i] = ''
#         E94010_df['Theta'].iloc[i] = ''
#         E94010_df['Eb'].iloc[i] = ''
#         E94010_df['nu'].iloc[i] = ''
#         E94010_df['G1.mes'].iloc[i] = ''
#         E94010_df['G1.mes.err'].iloc[i] = ''
#         E94010_df['G2.mes'].iloc[i] = ''
#         E94010_df['G2.mes.err'].iloc[i] = ''
#         E94010_df['G1F1.mes'].iloc[i] = ''
#         E94010_df['G1F1.mes.err'].iloc[i] = ''
#         E94010_df['G2F1.mes'].iloc[i] = ''
#         E94010_df['G2F1.mes.err'].iloc[i] = ''

# E94010_df.to_csv("publishedData.csv", index=False)

nd_df = pd.read_csv('NeutronData.csv')
nd_df = nd_df.iloc[74:108]
nd_df = nd_df.dropna(axis=1, how='all')
nd_df.to_csv('neutron_SLAC_E155.csv')

# E155df = pd.read_csv('neutron_SLAC_E155.csv')

# nd_df.loc[84:107, 'G1.mes'] = E155df['G1.mes'].values
# nd_df.loc[84:107, 'G1.mes.err'] = E155df['G1.mes.err'].values

# nd_df.to_csv('NeutronData.csv')