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

# nd_df = pd.read_csv('NeutronData.csv')
# nd_df = nd_df.iloc[74:108]
# nd_df = nd_df.dropna(axis=1, how='all')
# nd_df.to_csv('neutron_SLAC_E155.csv')

# E155df = pd.read_csv('neutron_SLAC_E155.csv')

# nd_df.loc[84:107, 'G1.mes'] = E155df['G1.mes'].values
# nd_df.loc[84:107, 'G1.mes.err'] = E155df['G1.mes.err'].values

# nd_df.to_csv('NeutronData.csv')

helium = pd.read_csv("threeHedata.csv")
neutron = pd.read_csv("NeutronData.csv")
def add_column(df,existing,new):
    insert_index = df.columns.get_loc(existing)
    df.insert(loc=insert_index, column=new, value=np.nan)

# add_column(neutron,'g2','dg1(sys)')
# add_column(neutron,'g2','dg1(tot)')
# add_column(neutron,'g1/F1','dg2(sys)')
# add_column(neutron,'g1/F1','dg2(tot)')
# add_column(neutron,'g2/F1','dg1/F1(sys)')
# add_column(neutron,'g2/F1','dg1/F1(tot)')
# add_column(neutron,'sigma_LT','dg2/F1(sys)')
# add_column(neutron,'sigma_LT','dg2/F1(tot)')
# add_column(neutron,'sigma_TT','dsigma_LT(sys)')
# add_column(neutron,'sigma_TT','dsigma_LT(tot)')
# add_column(neutron,'Experiment','dsigma_TT(sys)')
# add_column(neutron,'Experiment','dsigma_TT(tot)')
# helium.drop(helium.columns[0], axis=1, inplace=True)
# helium.replace(-1000.0,'', inplace=True)

# add_column(neutron,'g2/F1','A1')
# add_column(neutron,'g2/F1','dA1(stat)')
# add_column(neutron,'g2/F1','dA1(sys)')
# add_column(neutron,'g2/F1','dA1(tot)')

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

# mask = helium['Experiment'] == 'SLAC E142'
# helium.loc[mask, ['dg1(stat)', 'dg1(tot)']] = helium.loc[mask, ['dg1(tot)', 'dg1(stat)']].values

# mask2 = helium['Experiment'] == 'E97110'
# helium.loc[mask2, ['dg1(stat)', 'dg1(tot)']] = helium.loc[mask2, ['dg1(tot)', 'dg1(stat)']].values

# mask3 = helium['Experiment'] == 'E94010'
# helium.loc[mask3, ['dg1(stat)', 'dg1(tot)']] = helium.loc[mask3, ['dg1(tot)', 'dg1(stat)']].values

# mask = helium['Experiment'] == 'SLAC E142'
# helium.loc[mask, ['dg2(stat)', 'dg2(tot)']] = helium.loc[mask, ['dg2(tot)', 'dg2(stat)']].values

# def quadrature_sum(df, col1, col2, result, experiments):
#     mask = df['Experiment'].isin(experiments)
#     col1_numeric = pd.to_numeric(df.loc[mask, col1], errors='coerce')
#     col2_numeric = pd.to_numeric(df.loc[mask, col2], errors='coerce')
#     df.loc[mask, result] = np.sqrt(col1_numeric**2 + col2_numeric**2)

# experiments_to_update = ['Zheng']
# quadrature_sum(helium, 'dg1(stat)', 'dg1(sys)', 'dg1(tot)', experiments_to_update)
# print(helium['dg1(tot)'].sample(20))
# helium = helium.round(4)
# helium.to_csv("threeHedata.csv", index=False)

for experiment in helium['Experiment'].unique():
    df_subset = helium[helium['Experiment'] == experiment]
    df_subset = df_subset.dropna(axis=1, how='all')
    df_subset = df_subset.iloc[:, :-1]
    filename = f"{experiment}.csv"
    df_subset.to_csv(filename, index=False)