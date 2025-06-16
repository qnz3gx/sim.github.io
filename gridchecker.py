# %%
import pandas as pd
import numpy as np

def import_csv(file_path):
    return pd.read_csv(file_path)
# %%
# Load tables
tableP = import_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/table_P_sm0_ipol1_ipolres1_IA14_SF23_AC11_mod1.out")
tableN = import_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/table_N_sm0_ipol1_ipolres1_IA14_SF23_AC11_mod1.out")
tableD = import_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/table_D_sm3_ipol1_ipolres1_IA14_SF23_AC11_mod1.out")

# Strip spaces from column names
for df in [tableD, tableN, tableP]:
    df.columns = df.columns.str.strip()

# Create keys for alignment
for df in [tableD, tableN, tableP]:
    df['key'] = list(zip(df['Q2'], df['X']))

# Find shared Q2–X pairs
common_keys = set(tableD['key']) & set(tableN['key']) & set(tableP['key'])

# Filter all tables to common keys
tableD = tableD[tableD['key'].isin(common_keys)].reset_index(drop=True)
tableN = tableN[tableN['key'].isin(common_keys)].reset_index(drop=True)
tableP = tableP[tableP['key'].isin(common_keys)].reset_index(drop=True)

# Now drop 'key' column
for df in [tableD, tableN, tableP]:
    df.drop(columns='key', inplace=True)

# Apply Q² > 1 and W² > 4 mask to all three tables
mask = (tableD['Q2'] > 1) & (tableD['W2'] > 4)

# Apply the same mask to all three (since they're aligned now)
D = tableD[mask].reset_index(drop=True)
N = tableN[mask].reset_index(drop=True)
P = tableP[mask].reset_index(drop=True)

# Calculate difference
distance_F = D['F1'] - (N['F1'] + P['F1'])
distance_G = D['G1'] - (D['A1'] * D['F1'])

# Report where the difference is large
large_differencesF = distance_F[abs(distance_F) > 0.2]

print(max(large_differencesF))
print(max(distance_G),min(distance_G))

# %%
wD = 0.056
wDerr = 0.01

CompassP_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/CompassProton.csv"
CompassD_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/CompassDeuteron.csv"

CompassP = import_csv(CompassP_path)
CompassD = import_csv(CompassD_path)

for df in [CompassP,CompassD]:
    df['key'] = list(zip(df['Q2'],df['X']))

same_keys = set(CompassP['key']) & set(CompassD['key'])
CompassPsame = CompassP[CompassP['key'].isin(same_keys)].reset_index(drop=True)
CompassDsame = CompassD[CompassD['key'].isin(same_keys)].reset_index(drop=True)

for df in [CompassP,CompassD,CompassPsame,CompassDsame]:
    df.drop(columns='key', inplace=True)

def retrieve_f1(data_df,grid_df):
    f1_values = []
    for i in range(len(data_df)):
        target_X = data_df['X'].iloc[i]
        target_Q2 = data_df['Q2'].iloc[i]
        
        # Compute distances from all points in grid_df
        distances = np.sqrt((grid_df['X'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
        nearest_idx = distances.idxmin()
        
        # Get the corresponding F1 value
        f1_values.append(grid_df.loc[nearest_idx, 'F1'])
    return f1_values

#Get the F1, G1, and G1 error values for the deuteron
CompassD['Q2'] = CompassP['Q2']
CompassD['F1'] = retrieve_f1(CompassD,tableD)
CompassD['G1'] = CompassD['A1'] * CompassD['F1']
CompassD['G1.err'] = CompassD['A1.err']*CompassD['F1']

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
neutron_COMPASS['W'] = retrieve_w(neutron_COMPASS,tableN)
neutron_COMPASS['G1.mes'] = (1-1.5*wD)*CompassD['G1'] - CompassP['G1']
sigma=(1-1.5*wD)*CompassD['G1']*np.sqrt((1.5*wDerr/(1-1.5*wD))**2 + (CompassD['G1.err']/CompassD['G1'])**2)
neutron_COMPASS['G1.mes.err'] = np.sqrt((sigma)**2 + (CompassP['G1.err'])**2)

# %%
def retrieve_g1(data_df,grid_df):
    g1_values = []
    for i in range(len(data_df)):
        target_X = data_df['X'].iloc[i]
        target_Q2 = data_df['Q2'].iloc[i]
        
        # Compute distances from all points in grid_df
        distances = np.sqrt((grid_df['X'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
        nearest_idx = distances.idxmin()
        
        # Get the corresponding W value
        g1_values.append(grid_df.loc[nearest_idx, 'G1'])
    return g1_values

#looking for bad calculations
checking = retrieve_g1(neutron_COMPASS,tableN) - neutron_COMPASS['G1.mes']

#add direct calculation of G1n for row where X and Q2 were the same for G1d and p
Sameneut = (1-1.5*wD)*CompassDsame['G1'] - CompassPsame['G1']
sigmaSame=(1-1.5*wD)*CompassDsame['G1']*np.sqrt((1.5*wDerr/(1-1.5*wD))**2 + (CompassDsame['G1.err']/CompassDsame['G1'])**2)
erroneous = np.sqrt((sigma)**2 + (CompassPsame['G1.err'])**2)
neutron_COMPASS.loc[0,'G1.mes'] = Sameneut[0]
neutron_COMPASS.loc[0,'G1.mes.err'] = erroneous[0]
print(len(neutron_COMPASS))

# %%
spreadsheet = pd.DataFrame()
spreadsheet['X'] = neutron_COMPASS['X']
spreadsheet['Q2'] = neutron_COMPASS['Q2']
spreadsheet['A1p'] = CompassP['A1']
spreadsheet['A1p.err'] = CompassP['A1.err']
spreadsheet['G1p'] = CompassP['G1']
spreadsheet['G1p.err'] = CompassP['G1.err']
spreadsheet['F1p'] = retrieve_f1(CompassP,tableP) 
spreadsheet['G1n'] = neutron_COMPASS['G1.mes']
spreadsheet['G1n.err'] = neutron_COMPASS['G1.mes.err']
spreadsheet['F1n'] = retrieve_f1(neutron_COMPASS,tableN)
spreadsheet['A1n'] = spreadsheet['G1n']/spreadsheet['F1n']
spreadsheet['A1d'] = CompassD['A1']
spreadsheet['A1d.err'] = CompassD['A1.err']
spreadsheet['F1d'] = CompassD['F1']
spreadsheet['G1d'] = CompassD['G1']
spreadsheet['G1d.err'] = CompassD['G1.err']

spreadsheet.to_csv('COMPASS.csv',index=False)
# %%
# export the data to neutron data
nd_df = import_csv('NeutronData.csv')
print(nd_df.loc[349:365])

nd_df.loc[349:365, 'G1.mes'] = neutron_COMPASS['G1.mes'].values
nd_df.loc[349:365, 'G1.mes.err'] = neutron_COMPASS['G1.mes.err'].values
nd_df.loc[349:365, 'X'] = neutron_COMPASS['X'].values
nd_df.loc[349:365, 'Q2'] = neutron_COMPASS['Q2'].values
nd_df.loc[349:365, 'W'] = neutron_COMPASS['W'].values

nd_df.to_csv('NeutronData.csv',index=False)
# %%
