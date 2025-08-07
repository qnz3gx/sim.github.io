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

def replace_exp(main_df, exp_df, exp,rowa,rowb):
    common_cols = main_df.columns.intersection(exp_df.columns)
    main_df.loc[rowa:rowb, common_cols] = exp_df[common_cols].values
    main_df.loc[rowa:rowb,'Experiment'] = exp
    print(f"Replaced columns: {list(common_cols)}")
    return main_df

def quadrature_sum(df,col1,col2,result):
    mask = df[col1].notna() & df[col2].notna()
    df.loc[mask, result] = np.sqrt(df.loc[mask, col1]**2 + df.loc[mask, col2]**2)
    return df

# cj15nlo = pd.read_csv("deuteron_CJ15nlo.csv")
# ct18nnlo = pd.read_csv("deuteron_CT18NNLO.csv")

# compass = (cj15nlo + ct18nnlo)/2
# compass['x'] = cj15nlo['x']
# compass['Q2'] = cj15nlo['Q2']
# compass['dg1(stat)'] = 1/2 * np.sqrt(cj15nlo['dg1(stat)'].values ** 2 + ct18nnlo['dg1(stat)'].values ** 2)
# compass['dg1(sys)'] = 1/2 * np.sqrt(cj15nlo['dg1(sys)'].values ** 2 + ct18nnlo['dg1(sys)'].values ** 2)

# three = [cj15nlo,ct18nnlo]

def maxerr(datasets,column):
    maximum_error = []
    for i in range(len(datasets[0])):
        maximum = np.max([abs(datasets[0].iloc[i][column]-datasets[1].iloc[i][column]), abs(datasets[1].iloc[i][column]-datasets[0].iloc[i][column])])
        maximum_error.append(maximum/2)
    return maximum_error

# jamn=pd.read_csv('neutron_JAM22.csv')
# quadrature_sum(jamn,'dg1(stat)','dg1(sys)','dg1(tot)')
# quadrature_sum(jamn,'dg1/F1(stat)','dg1/F1(sys)','dg1/F1(tot)')
# jamn=jamn.round(4)
# jamn.to_csv('neutron_JAM22.csv', index=False)
# neutron=replace_exp(neutron,jamn,'COMPASS(JAM22)',394,408)
# neutron.to_csv("NeutronData.csv", index=False)

# jamp=pd.read_csv('proton_JAM22.csv')
# quadrature_sum(jamp,'dg1(stat)','dg1(sys)','dg1(tot)')
# quadrature_sum(jamp,'dA1(stat)','dA1(sys)','dA1(tot)')
# jamp=jamp.round(4)
# jamp.to_csv('proton_JAM22.csv', index=False)
# proton=replace_exp(proton,jamp,'COMPASS(JAM22)',3029,3043)
# proton.to_csv("ProtonData.csv", index=False)

# jamd=pd.read_csv('deuteron_JAM22.csv')
# quadrature_sum(jamd,'dg1(stat)','dg1(sys)','dg1(tot)')
# quadrature_sum(jamd,'dA1(stat)','dA1(sys)','dA1(tot)')
# jamd=jamd.round(4)
# jamd.to_csv('deuteron_JAM22.csv', index=False)
# deuteron=replace_exp(deuteron,jamd,'COMPASS(JAM22)',7967,7981)
# deuteron.to_csv("DeuteronData.csv", index=False)

hermes = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/threeHe_Hermes.csv")
quadrature_sum(hermes, 'dA1(stat)', 'dA1(sys)', 'dA1(tot)')
hermes = hermes.round(4)
hermes.to_csv("threeHe_Hermes.csv",index=False)
helium = replace_exp(helium,hermes,'HERMES',107,115)
helium.to_csv("threeHedata.csv",index=False)