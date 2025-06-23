# %%
import pandas as pd
import numpy as np
# %%
# Load tables
# tableD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/deu_F1.csv") #CJ15nlo
# tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CJ15nlon_F1.csv") #CJ15nlo
# tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CJ15nlop_F1.csv")

# tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CT18NNLOn_F1.csv") #CT18NNLO
# tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/CT18NNLOp_F1.csv")

tableD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/table_D_sm3_ipol1_ipolres1_IA14_SF23_AC11_mod1.out") #Zheng
tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/table_N_sm0_ipol1_ipolres1_IA14_SF23_AC11_mod1.out")
tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/table_P_sm0_ipol1_ipolres1_IA14_SF23_AC11_mod1.out")

CompassP_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/CompassProton.csv"
CompassD_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/CompassDeuteron.csv"

CompassP = pd.read_csv(CompassP_path)
CompassD = pd.read_csv(CompassD_path)

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
CompassD['G1'] = CompassD['A1'] * CompassD['F1']
CompassD['G1.err'] = CompassD['A1.err'] * CompassD['F1']
CompassD['g1.err'] = CompassD['a1.err'] * CompassD['F1']

#Get F1, G1, and G1 error values for the proton
CompassP['F1'] = retrieve_f1(CompassP,tableP)
CompassP['G1'] = CompassP['A1'] * CompassP['F1']
CompassP['G1.err'] = CompassP['A1.err'] * CompassP['F1']
CompassP['g1.err'] = CompassP['a1.err'] * CompassP['F1']

#Create a neutron dataframe
neutron_COMPASS = pd.DataFrame()
neutron_COMPASS['Q2'] = CompassP['Q2']
neutron_COMPASS['X'] = (CompassP['X'] + CompassD['X'])/2

#Get G1 and G1 error for the neutron
neutron_COMPASS['G1.mes'] = (1-1.5*wD)*CompassD['G1'] - CompassP['G1']
stat=(1-1.5*wD)*CompassD['G1']*np.sqrt((1.5*wDerr/(1-1.5*wD))**2 + (CompassD['G1.err']/CompassD['G1'])**2)
sys=(1-1.5*wD)*CompassD['G1']*np.sqrt((1.5*wDerr/(1-1.5*wD))**2 + (CompassD['g1.err']/CompassD['G1'])**2)
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
spreadsheet['A1n'] = spreadsheet['G1n']/spreadsheet['F1n']
spreadsheet['dA1n(stat)'] = spreadsheet['dG1n(stat)']/spreadsheet['F1n']
spreadsheet['dA1n(sys)'] = spreadsheet['dG1n(sys)']/spreadsheet['F1n']

spreadsheet = spreadsheet.round(4)
spreadsheet.to_csv('COMPASS_recalculated_Zheng.csv',index=False)
# %%
# export separate neutron df
neutron = pd.DataFrame()
neutron['x'] = neutron_COMPASS['X']
neutron['Q2'] = neutron_COMPASS['Q2']
neutron['g1'] = neutron_COMPASS['G1.mes']
neutron['dg1(stat)'] = neutron_COMPASS['G1.mes.err']
neutron['dg1(sys)'] = neutron_COMPASS['g1.mes.err']
neutron['F1'] = retrieve_f1(neutron_COMPASS,tableN)
neutron['g1/F1'] = spreadsheet['G1n']/spreadsheet['F1n']
neutron['dg1/F1(stat)'] = spreadsheet['dG1n(stat)']/spreadsheet['F1n']
neutron['dg1/F1(sys)'] = spreadsheet['dG1n(sys)']/spreadsheet['F1n']

neutron = neutron.round(4)
neutron.to_csv('neutron_COMPASS.csv',index=False)
# %%
