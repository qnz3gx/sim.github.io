import pandas as pd
import numpy as np
import plotly.express as px

helium = pd.read_csv("threeHedata.csv")

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
        filename = f"neutron_{experiment}.csv" #don't forget to change this
        df_subset.to_csv(filename, index=False)

def replace_exp(main_df, exp_df, exp):
    common_cols = main_df.columns.intersection(exp_df.columns)
    main_df.loc[3:20, common_cols] = exp_df[common_cols].values   
    print(f"Replaced columns: {list(common_cols)}")
    return main_df

def quadrature_sum(df,col1,col2,result):
    mask = df[col1].notna() & df[col2].notna()
    df.loc[mask, result] = np.sqrt(df.loc[mask, col1]**2 + df.loc[mask, col2]**2)
    return df

X_target = 0.3
Q2_target = 5.1

tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/p_F1.csv") #used CT18NNLO
tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/n_F1.csv")
tableD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/d_F1.csv")

def retrieve_f1(grid_df, x_target, Q2_target):
    distances = np.sqrt((grid_df['x'] - x_target)**2 + (grid_df['Q2'] - Q2_target)**2)
    nearest_idx = distances.idxmin()
    return grid_df.loc[nearest_idx, 'F1']

def retrieve_g1(grid_df, x_target, Q2_target):
    distances = np.sqrt((grid_df['x'] - x_target)**2 + (grid_df['Q2'] - Q2_target)**2)
    nearest_idx = distances.idxmin()
    return grid_df.loc[nearest_idx, 'g1']

# Call with explicit coordinates
f1p = retrieve_f1(tableP, X_target, Q2_target)
f1n = retrieve_f1(tableN, X_target, Q2_target)
g1p = retrieve_g1(tableP, X_target, Q2_target)
g1n = retrieve_g1(tableN, X_target, Q2_target)

print(f"F1p: {f1p:.4f}, g1p: {g1p:.4f}), F1n: {f1n:.4f}, g1n: {g1n:.4f}")