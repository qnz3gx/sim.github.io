# %%
import pandas as pd
import numpy as np
# %%
# Load tables

#CJ15nlo
# tableD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CJ15_F1d_grid.csv")
# tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CJ15_F1n_grid.csv")
# tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CJ15_F1p_grid.csv")

#CT18NNLO
tableD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CT18_F1d_grid.csv")
tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CT18_F1n_grid.csv")
tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CT18_F1p_grid.csv")

#JAM22
# tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/JAM22_F1p_grid.csv")
# tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/JAM22_F1n_grid.csv")
# tableD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/JAM22_F1d_grid.csv")

#Original COMPASS Data
CompassP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/CompassProton.csv")
CompassD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/CompassDeuteron.csv")

# Strip spaces from column names
for df in [tableD, tableN, tableP]:
    df.columns = df.columns.str.strip()

# %%
wD = 0.056
wDerr = 0.01

def retrieve_f1(data_df,grid_df):
    f1_values = []
    for i in range(len(data_df)):
        target_X = data_df['X'].iloc[i]
        target_Q2 = data_df['Q2'].iloc[i]
        
        # Compute distances from all points in grid_df
        distances = np.sqrt((grid_df['x'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
        nearest_idx = distances.idxmin()

        # Get the corresponding F1 value
        f1_values.append(grid_df.loc[nearest_idx, 'F1'])
    return f1_values

def retrieve_f1_err(data_df,grid_df):
    f1_err_values = []
    for i in range(len(data_df)):
        target_X = data_df['X'].iloc[i]
        target_Q2 = data_df['Q2'].iloc[i]
        
        # Compute distances from all points in grid_df
        distances = np.sqrt((grid_df['x'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
        nearest_idx = distances.idxmin()

        # Get the corresponding F1 value
        f1_err_values.append(grid_df.loc[nearest_idx, 'dF1(tot)'])
    return f1_err_values

def retrieve_w(data_df,grid_df):
    w_values = []
    for i in range(len(data_df)):
        target_X = data_df['X'].iloc[i]
        target_Q2 = data_df['Q2'].iloc[i]
        
        # Compute distances from all points in grid_df
        distances = np.sqrt((grid_df['X'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
        nearest_idx = distances.idxmin()
        
        # Get the corresponding W value
        w_values.append(np.sqrt(grid_df.loc[nearest_idx, 'W2']))
    return w_values

def retrieve_g1(data_df,grid_df):
    g1_values = []
    for i in range(len(data_df)):
        target_X = data_df['X'].iloc[i]
        target_Q2 = data_df['Q2'].iloc[i]
        
        # Compute distances from all points in grid_df
        distances = np.sqrt((grid_df['x'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
        nearest_idx = distances.idxmin()
        
        # Get the corresponding g1 value
        g1_values.append(grid_df.loc[nearest_idx, 'g1'])
    return g1_values

# %%
#Get the F1, G1, and G1 error values for the deuteron
CompassD['Q2'] = CompassP['Q2'].values
CompassD['F1'] = retrieve_f1(CompassD,tableD)
CompassD['F1.err'] = retrieve_f1_err(CompassD, tableD)
CompassD['G1'] = CompassD['A1'] * CompassD['F1']
#error for division: g1*sqrt((a1err/a1)^2 + (f1err/f1)^2)
CompassD['G1.err'] = np.abs(CompassD['G1']) * np.sqrt((CompassD['A1.err'].values/CompassD['A1'].values)**2 + (CompassD['F1.err'].values/CompassD['F1'].values)**2)
CompassD['g1.err'] = np.abs(CompassD['G1']) * np.sqrt((CompassD['a1.err'].values/CompassD['A1'].values)**2 + (CompassD['F1.err'].values/CompassD['F1'].values)**2)

#Get F1, G1, and G1 error values for the proton
CompassP['F1'] = retrieve_f1(CompassP,tableP)
CompassP['F1.err'] = retrieve_f1_err(CompassP, tableP)
CompassP['G1'] = CompassP['A1'] * CompassP['F1']
CompassP['G1.err'] = CompassP['G1'] * np.sqrt((CompassP['A1.err'].values/CompassP['A1'].values)**2 + (CompassP['F1.err'].values/CompassP['F1'].values)**2)
CompassP['g1.err'] = CompassP['G1'] * np.sqrt((CompassP['a1.err'].values/CompassP['A1'].values)**2 + (CompassP['F1.err'].values/CompassP['F1'].values)**2)

#Create a neutron dataframe
neutron_COMPASS = pd.DataFrame()
neutron_COMPASS['Q2'] = CompassP['Q2']
neutron_COMPASS['X'] = (CompassP['X'] + CompassD['X'])/2

#Get G1 and G1 error for the neutron
neutron_COMPASS['G1.mes'] = CompassD['G1'] / (1-1.5*wD) - CompassP['G1']
stat=CompassD['G1']/(1-1.5*wD) * np.sqrt((1.5*wDerr/(1-1.5*wD))**2 + (CompassD['G1.err']/CompassD['G1'])**2)
sys=CompassD['G1']/(1-1.5*wD) * np.sqrt((1.5*wDerr/(1-1.5*wD))**2 + (CompassD['g1.err']/CompassD['G1'])**2)
stat = stat.fillna(0)
sys = sys.fillna(0)
neutron_COMPASS['G1.mes.err'] = np.sqrt((stat)**2 + (CompassP['G1.err'])**2)
neutron_COMPASS['g1.mes.err'] = np.sqrt((sys)**2 + (CompassP['g1.err'])**2)
# %%
spreadsheet = pd.DataFrame()
spreadsheet['x'] = neutron_COMPASS['X']
spreadsheet['Q2'] = neutron_COMPASS['Q2']

spreadsheet['G1p'] = CompassP['G1']
spreadsheet['dG1p(stat)'] = CompassP['G1.err']
spreadsheet['dG1p(sys)'] = CompassP['g1.err']
spreadsheet['F1p'] = CompassP['F1']
spreadsheet['A1p'] = CompassP['A1']
spreadsheet['dA1p(stat)'] = CompassP['A1.err']
spreadsheet['dA1p(sys)'] = CompassP['a1.err']

spreadsheet['G1d'] = CompassD['G1']
spreadsheet['dG1d(stat)'] = CompassD['G1.err']
spreadsheet['dG1d(sys)'] = CompassD['g1.err']
spreadsheet['F1d'] = CompassD['F1']
spreadsheet['A1d'] = CompassD['G1']/spreadsheet['F1d']
spreadsheet['dA1d(stat)'] = CompassD['G1.err']/spreadsheet['F1d']
spreadsheet['dA1d(sys)'] = CompassD['g1.err']/spreadsheet['F1d']

spreadsheet['G1n'] = neutron_COMPASS['G1.mes']
spreadsheet['dG1n(stat)'] = neutron_COMPASS['G1.mes.err']
spreadsheet['dG1n(sys)'] = neutron_COMPASS['g1.mes.err']
spreadsheet['F1n'] = retrieve_f1(neutron_COMPASS,tableN)
spreadsheet['dF1n(tot)'] = retrieve_f1_err(neutron_COMPASS, tableN)
spreadsheet['g1/F1n'] = spreadsheet['G1n']/spreadsheet['F1n']
spreadsheet['dg1/F1n(stat)'] = spreadsheet['g1/F1n'] * np.sqrt((spreadsheet['dG1n(stat)'].values/spreadsheet['G1n'].values) ** 2 + (spreadsheet['dF1n(tot)'].values/spreadsheet['F1n'].values) ** 2) #spreadsheet['dG1n(stat)']/spreadsheet['F1n']
spreadsheet['dg1/F1n(sys)'] = spreadsheet['g1/F1n'] * np.sqrt((spreadsheet['dG1n(sys)'].values/spreadsheet['G1n'].values) ** 2 + (spreadsheet['dF1n(tot)'].values/spreadsheet['F1n'].values) ** 2) #spreadsheet['dG1n(sys)']/spreadsheet['F1n']

spreadsheet = spreadsheet.round(4)
# spreadsheet.to_csv('ogPDg1F1_JAM22.csv',index=False)
# %%
# export separate neutron df
neutron = pd.DataFrame()
neutron['x'] = neutron_COMPASS['X']
neutron['Q2'] = neutron_COMPASS['Q2']
neutron['g1'] = neutron_COMPASS['G1.mes']
neutron['dg1(stat)'] = neutron_COMPASS['G1.mes.err']
neutron['dg1(sys)'] = neutron_COMPASS['g1.mes.err']
neutron['F1'] = retrieve_f1(neutron_COMPASS,tableN)
neutron['dF1(tot)'] = spreadsheet['dF1n(tot)']
neutron['g1/F1'] = spreadsheet['G1n']/spreadsheet['F1n']
neutron['dg1/F1(stat)'] = spreadsheet['dG1n(stat)']/spreadsheet['F1n']
neutron['dg1/F1(sys)'] = spreadsheet['dG1n(sys)']/spreadsheet['F1n']

neutron = neutron.round(4)
neutron.to_csv('neutron_CT18.csv',index=False)
# %%
#export separate proton df
proton = pd.DataFrame()
proton['x'] = spreadsheet['x']
proton['Q2'] = spreadsheet['Q2']
proton['g1'] = spreadsheet['G1p']
proton['dg1(stat)'] = spreadsheet['dG1p(stat)']
proton['dg1(sys)'] = spreadsheet['dG1p(sys)']
proton['F1'] = spreadsheet['F1p']
proton['dF1(tot)'] = CompassP['F1.err']
proton['A1'] = spreadsheet['A1p']
proton['dA1(stat)'] = spreadsheet['dA1p(stat)']
proton['dA1(sys)'] = spreadsheet['dA1p(sys)']

proton = proton.round(4)
proton.to_csv('proton_CT18.csv',index=False)
# %%
#export separate proton df
deuteron = pd.DataFrame()
deuteron['x'] = spreadsheet['x']
deuteron['Q2'] = spreadsheet['Q2']
deuteron['g1'] = spreadsheet['G1d']
deuteron['dg1(stat)'] = spreadsheet['dG1d(stat)']
deuteron['dg1(sys)'] = spreadsheet['dG1d(sys)']
deuteron['F1'] = spreadsheet['F1d']
deuteron['dF1(tot)'] = CompassD['F1.err']
deuteron['A1'] = spreadsheet['A1d']
deuteron['dA1(stat)'] = spreadsheet['dA1d(stat)']
deuteron['dA1(sys)'] = spreadsheet['dA1d(sys)']

deuteron = deuteron.round(4)
deuteron.to_csv('deuteron_CT18.csv',index=False)
# %%