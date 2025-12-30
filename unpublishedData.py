import pandas as pd
import numpy as np
import plotly.express as px

helium = pd.read_csv("threeHedata.csv")
neutron = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/NeutronData.csv")
deuteron = pd.read_csv("DeuteronData.csv")
proton = pd.read_csv("ProtonData.csv")

def add_column(df,existing,new):
    insert_index = df.columns.get_loc(existing)
    df.insert(loc=insert_index, column=new, value=np.nan)

# def quadrature_sum(df, col1, col2, result, experiments):
#     mask = df['Experiment'].isin(experiments)
#     col1_numeric = pd.to_numeric(df.loc[mask, col1], errors='coerce')
#     col2_numeric = pd.to_numeric(df.loc[mask, col2], errors='coerce')
#     df.loc[mask, result] = np.sqrt(col1_numeric**2 + col2_numeric**2)

def sort(df, particle):
    for experiment in df['Experiment'].unique():
        df_subset = df[df['Experiment'] == experiment]
        df_subset = df_subset.dropna(axis=1, how='all')
        df_subset = df_subset.iloc[:, :-1]
        filename = f"{particle}_{experiment}.csv"
        df_subset.to_csv(filename, index=False)

def replace_exp(main_df, exp_df, exp, rowa, rowb):
    common_cols = main_df.columns.intersection(exp_df.columns)
    main_df.loc[rowa:rowb, common_cols] = exp_df[common_cols].values
    main_df.loc[rowa:rowb,'Experiment'] = exp
    print(f"Replaced columns: {list(common_cols)}")
    return main_df

def quadrature_sum(df,col1,col2,result):
    mask = df[col1].notna() & df[col2].notna()
    df.loc[mask, result] = np.sqrt(df.loc[mask, col1]**2 + df.loc[mask, col2]**2)
    return df

def maxerr(datasets,column):
    maximum_error = []
    for i in range(len(datasets[0])):
        maximum = np.max([abs(datasets[0].iloc[i][column]-datasets[1].iloc[i][column]), abs(datasets[1].iloc[i][column]-datasets[0].iloc[i][column])])
        maximum_error.append(maximum/2)
    return maximum_error

eg1a = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/proton_CLAS_EG1a.csv")
eg1b = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/proton_CLAS_EG1b.csv")

proton = replace_exp(proton,eg1a,'CLAS_EG1a',6576,6738)
print('eg1a')
proton = replace_exp(proton,eg1b,'CLAS_EG1b',47,5903)
print('eg1b')

proton.to_csv("ProtonData.csv",index=False)