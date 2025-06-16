# %%
import pandas as pd
import numpy as np

dataset = 'neutron_SLAC_E155'
skip=0

#upload data to dataframes
def import_csv(file_path, lines):
    data = pd.read_csv(file_path, skiprows=lines, index_col=False)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.columns = data.columns.str.strip()
    return data

grid_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/problemchild.csv"
G1_path = f"/Users/scarlettimorse/PycharmProjects/sim.github.io/{dataset}.csv"
grid_df = import_csv(grid_path, 0)
data_df = import_csv(G1_path, skip)
data_df = data_df[~data_df['G1F1.mes'].isnull()]
# %%
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
    print(f"F1_IpQE = {F1_IpQE_value}")
    print(f"G1F1.mes = {data_df['G1F1.mes'].iloc[i]}")
    print(f"Grid row:\n{nearest_idx}")
    print("-" * 50)

data_df['G1.mes'] = G1calc
data_df['G1.mes.err'] = G1errcalc
# %%
data_df.to_csv(f'{dataset}.csv')