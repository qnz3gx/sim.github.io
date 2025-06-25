import pandas as pd
import numpy as np
import plotly.express as px

helium = pd.read_csv("threeHedata.csv")
neutron = pd.read_csv("NeutronData.csv")
deuteron = pd.read_csv("DeuteronData.csv")
proton = pd.read_csv("ProtonData.csv")

def add_column(df,existing,new):
    insert_index = df.columns.get_loc(existing)
    df.insert(loc=insert_index, column=new, value=np.nan)

def quadrature_sum(df, col1, col2, result, experiments):
    mask = df['Experiment'].isin(experiments)
    col1_numeric = pd.to_numeric(df.loc[mask, col1], errors='coerce')
    col2_numeric = pd.to_numeric(df.loc[mask, col2], errors='coerce')
    df.loc[mask, result] = np.sqrt(col1_numeric**2 + col2_numeric**2)

def sort(df):
    for experiment in df['Experiment'].unique():
        df_subset = df[df['Experiment'] == experiment]
        df_subset = df_subset.dropna(axis=1, how='all')
        df_subset = df_subset.iloc[:, :-1]
        filename = f"deuteron_{experiment}.csv" #don't forget to change this
        df_subset.to_csv(filename, index=False)

def replace_exp(main_df, exp_df, exp):
    common_cols = main_df.columns.intersection(exp_df.columns)
    main_df.loc[379:393, common_cols] = exp_df[common_cols].values
    main_df.loc[379:393,'Experiment'] = exp
    print(f"Replaced columns: {list(common_cols)}")
    return main_df

# def quadrature_sum(df,col1,col2,result):
#     mask = df[col1].notna() & df[col2].notna()
#     df.loc[mask, result] = np.sqrt(df.loc[mask, col1]**2 + df.loc[mask, col2]**2)
#     return df

# X_target = 0.3
# Q2_target = 5.1

# tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/p_F1.csv") #used CT18NNLO
# tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/n_F1.csv")
# tableD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/d_F1.csv")

# def retrieve_f1(grid_df, x_target, Q2_target):
#     distances = np.sqrt((grid_df['x'] - x_target)**2 + (grid_df['Q2'] - Q2_target)**2)
#     nearest_idx = distances.idxmin()
#     return grid_df.loc[nearest_idx, 'F1']

# def retrieve_g1(grid_df, x_target, Q2_target):
#     distances = np.sqrt((grid_df['x'] - x_target)**2 + (grid_df['Q2'] - Q2_target)**2)
#     nearest_idx = distances.idxmin()
#     return grid_df.loc[nearest_idx, 'g1']

# # Call with explicit coordinates
# f1p = retrieve_f1(tableP, X_target, Q2_target)
# f1n = retrieve_f1(tableN, X_target, Q2_target)
# g1p = retrieve_g1(tableP, X_target, Q2_target)
# g1n = retrieve_g1(tableN, X_target, Q2_target)

# print(f"F1p: {f1p:.4f}, g1p: {g1p:.4f}), F1n: {f1n:.4f}, g1n: {g1n:.4f}")

# zheng = pd.read_csv("neutron_COMPzheng.csv")
# cj15nlo = pd.read_csv("neutron_CJ15nlo.csv")
# ct18nnlo = pd.read_csv("neutron_CT18NNLO.csv")

# compass = (zheng + cj15nlo + ct18nnlo)/3
# compass['x'] = zheng['x']
# compass['Q2'] = zheng['Q2']
# compass['dg1(stat)'] = 1/3 * np.sqrt(zheng['dg1(stat)'].values ** 2 + cj15nlo['dg1(stat)'].values ** 2 + ct18nnlo['dg1(stat)'].values ** 2)
# compass['dg1(sys)'] = 1/3 * np.sqrt(zheng['dg1(sys)'].values ** 2 + cj15nlo['dg1(sys)'].values ** 2 + ct18nnlo['dg1(sys)'].values ** 2)

# three = [zheng,cj15nlo,ct18nnlo]

# def maxerr(datasets,column):
#     maximum_error = []
#     for i in range(len(datasets[0])):
#         maximum = np.max([abs(datasets[0].iloc[i][column]-datasets[1].iloc[i][column]), abs(datasets[1].iloc[i][column]-datasets[2].iloc[i][column]), abs(datasets[2].iloc[i][column]-datasets[0].iloc[i][column]-datasets[1].iloc[i][column])])
#         maximum_error.append(maximum/2)
#     return maximum_error

# compass['dg1(model)'] = maxerr(three,'g1')
# compass['dg1/F1(model)'] = maxerr(three,'g1/F1')

# compass['dg1(tot)'] = np.sqrt((compass['dg1(stat)'].values ** 2) + (compass['dg1(sys)'].values ** 2) + (compass['dg1(model)'].values ** 2))
# compass['dg1/F1(tot)'] = np.sqrt((compass['dg1/F1(stat)'].values ** 2) + (compass['dg1/F1(sys)'].values ** 2) + (compass['dg1/F1(model)'].values ** 2))

# replace_exp(neutron,compass, "COMPASS")
# print(neutron.tail())

# neutron = neutron.round(4)
# neutron.to_csv('NeutronData.csv', index=False)

# eg4 = pd.read_csv('proton_CLAS_EG4.csv')
# eg4['Experiment'] = 'CLAS_EG4'
# proton = pd.concat([proton,eg4])

# sort(deuteron)

compass = pd.read_csv('proton_COMPASS.csv')
compass=compass.drop(columns='dA1(model)')
proton.loc[proton['Experiment'] == 'COMPASS', 'dA1(model)'] = np.nan
proton=proton.dropna(axis=1, how='all')
compass.to_csv('proton_COMPASS.csv')
proton.to_csv('ProtonData.csv')