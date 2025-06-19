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