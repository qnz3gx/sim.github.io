# %%
import pandas as pd
import numpy as np
# %%
# Load tables
tableP = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/table_P_sm0_ipol1_ipolres1_IA14_SF23_AC11_mod1.out") #used CT18NNLO
tableN = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/table_N_sm0_ipol1_ipolres1_IA14_SF23_AC11_mod1.out")
tableD = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/table_D_sm3_ipol1_ipolres1_IA14_SF23_AC11_mod1.out") #used CJ15nlo

# Strip spaces from column names
for df in [tableD, tableN, tableP]:
    df.columns = df.columns.str.strip()

# Calculate difference
distance_F = tableD['F1'] - (tableN['F1'] + tableP['F1'])
distance_G = tableD['g1'] - (tableD['A1'] * tableD['F1'])

# Report where the difference is large
large_differencesF = distance_F[abs(distance_F) > 0.2]

#print(max(large_differencesF))
#print(max(distance_G),min(distance_G))

# %%
wD = 0.056
wDerr = 0.01

CompassP_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/CompassProton.csv"
CompassD_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/CompassDeuteron.csv"

CompassP = pd.read_csv(CompassP_path)
CompassD = pd.read_csv(CompassD_path)

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

#Get the F1, G1, and G1 error values for the deuteron
CompassD['Q2'] = CompassP['Q2'].values
CompassD['F1'] = retrieve_f1(CompassD,tableD)
CompassD['G1'] = CompassD['A1'] * CompassD['F1']
CompassD['G1.err'] = CompassD['A1.err']*CompassD['F1']
CompassD['g1.err'] = CompassD['a1.err']*CompassD['F1']

# %%
#Create a neutron dataframe
neutron_COMPASS = pd.DataFrame()
neutron_COMPASS['Q2'] = CompassP['Q2']
neutron_COMPASS['X'] = (CompassP['X'] + CompassD['X'])/2

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

#Get W, G1, and G1 error for the neutron
#neutron_COMPASS['W'] = retrieve_w(neutron_COMPASS,tableN)
neutron_COMPASS['G1.mes'] = (1-1.5*wD)*CompassD['G1'] - CompassP['G1']
sigma=(1-1.5*wD)*CompassD['G1']*np.sqrt((1.5*wDerr/(1-1.5*wD))**2 + (CompassD['G1.err']/CompassD['G1'])**2)
Sigma=(1-1.5*wD)*CompassD['G1']*np.sqrt((1.5*wDerr/(1-1.5*wD))**2 + (CompassD['g1.err']/CompassD['G1'])**2)
neutron_COMPASS['G1.mes.err'] = np.sqrt((sigma)**2 + (CompassP['G1.err'])**2)
neutron_COMPASS['g1.mes.err'] = np.sqrt((Sigma)**2 + (CompassP['g1.err'])**2)

# %%
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

#looking for bad calculations
checking = retrieve_g1(neutron_COMPASS,tableN) - neutron_COMPASS['G1.mes']
prop = retrieve_g1(neutron_COMPASS,tableN)/neutron_COMPASS['G1.mes']

# %%
spreadsheet = pd.DataFrame()
spreadsheet['x'] = neutron_COMPASS['X']
spreadsheet['Q2'] = neutron_COMPASS['Q2']
spreadsheet['A1p'] = CompassP['A1']
spreadsheet['dA1p(stat)'] = CompassP['A1.err']
spreadsheet['dA1p(sys)'] = CompassP['a1.err']
spreadsheet['G1p'] = CompassP['G1']
spreadsheet['dG1p(stat)'] = CompassP['G1.err']
spreadsheet['dG1p(sys)'] = CompassP['g1.err']
spreadsheet['F1p'] = retrieve_f1(CompassP,tableP) 
spreadsheet['G1n'] = neutron_COMPASS['G1.mes']
spreadsheet['dG1n(stat)'] = neutron_COMPASS['G1.mes.err']
spreadsheet['dG1n(sys)'] = neutron_COMPASS['g1.mes.err']
spreadsheet['F1n'] = retrieve_f1(neutron_COMPASS,tableN)
spreadsheet['A1n'] = spreadsheet['G1n']/spreadsheet['F1n']
spreadsheet['F1d'] = retrieve_f1(neutron_COMPASS, tableD)
spreadsheet['G1d'] = CompassD['G1']
spreadsheet['dG1d(stat)'] = CompassD['G1.err']
spreadsheet['dG1d(sys)'] = CompassD['g1.err']
spreadsheet['A1d'] = CompassD['G1']/spreadsheet['F1d']
spreadsheet['dA1d(stat)'] = CompassD['G1.err']/spreadsheet['F1d']
spreadsheet['dA1d(sys)'] = CompassD['g1.err']/spreadsheet['F1d']

spreadsheet = spreadsheet.round(4)
spreadsheet.to_csv('COMPASS.csv',index=False)
# %%
# export the data to neutron data
nd_df = pd.read_csv('NeutronData.csv')
print(nd_df.loc[349:365])

nd_df.loc[349:365, 'G1.mes'] = neutron_COMPASS['G1.mes'].values
nd_df.loc[349:365, 'G1.mes.err'] = neutron_COMPASS['G1.mes.err'].values
nd_df.loc[349:365, 'X'] = neutron_COMPASS['X'].values
nd_df.loc[349:365, 'Q2'] = neutron_COMPASS['Q2'].values
nd_df.loc[349:365, 'W'] = neutron_COMPASS['W'].values

#nd_df.to_csv('NeutronData.csv',index=False)
# %%
