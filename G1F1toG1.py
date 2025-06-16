import pandas as pd
import numpy as np

dataset = 'neutron_SLAC_E155'
skip=11

#upload data to dataframes
def import_csv(file_path, lines):
    data = pd.read_csv(file_path, skiprows=lines)
    return data

grid_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/XZ_table_3He_JAM_smeared_kpsv_onshell_ipol1_ipolres1_IA14_SF23_AC11_mod.csv"
G1_path = f"/Users/scarlettimorse/PycharmProjects/sim.github.io/{dataset}.csv"
grid_df = import_csv(grid_path, 0)
data_df = import_csv(G1_path, 0)

G1calc = []
G1errcalc = []
for i in range(len(data_df)):
    target_X = data_df['X'].iloc[i]
    target_Q2 = data_df['Q2'].iloc[i]
    
    # Compute distances from all points in grid_df
    distances = np.sqrt((grid_df['X'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
    nearest_idx = distances.idxmin()
    
    # Get the corresponding F1_IpQE value
    F1_IpQE_value = grid_df.loc[nearest_idx, 'F1_IpQE']
    G1calc.append(data_df['G1F1.mes'].iloc[i] * F1_IpQE_value)
    G1errcalc.append(data_df['G1F1.mes.err'].iloc[i] * F1_IpQE_value)

data_df['G1.mes'] = G1calc
data_df['G1.mes.err'] = G1errcalc

data_df.to_csv(f'{dataset}.csv')