# %%
import pandas as pd
import numpy as np

dataset = 'neutron_Flay'
skip=0

#upload data to dataframes
def import_csv(file_path, lines):
    data = pd.read_csv(file_path, skiprows=lines, index_col=False)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.columns = data.columns.str.strip()
    return data

grid_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/problemchild.csv"
G1_path = f"/Users/scarlettimorse/PycharmProjects/sim.github.io/{dataset}.csv"
grid = import_csv(grid_path, 0)
data_df = import_csv(G1_path, skip)
data_df = data_df[~data_df['g1/F1'].isnull()]
print(data_df.dtypes)
print(data_df[['x', 'Q2']].head())
print(data_df[['x', 'Q2']].isnull().sum())
# %%
G1calc = []
G1errcalc = []
G1syscalc = []
for i in range(len(data_df)):
    target_X = data_df['x'].iloc[i]
    target_Q2 = data_df['Q2'].iloc[i]

    grid_df = grid.dropna(subset=['X', 'Q2', 'F1_IpQE'])
    if grid_df.empty:
        print(f"Warning: grid_df has no valid rows at index {i}")
        G1calc.append(np.nan)
        G1errcalc.append(np.nan)
        G1syscalc.append(np.nan)
        continue
    
    # Compute distances from all points in grid_df
    distances = np.sqrt((grid_df['X'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
    
    if distances.isnull().all():
        print(f"Warning: All distances are NaN at index {i}")
        G1calc.append(np.nan)
        G1errcalc.append(np.nan)
        G1syscalc.append(np.nan)
        continue

    nearest_idx = distances.idxmin()

    # Get the corresponding F1_IpQE value
    F1_IpQE_value = grid_df.loc[nearest_idx, 'F1_IpQE']
    G1calc.append(data_df['g1/F1'].iloc[i] * F1_IpQE_value)
    G1errcalc.append(data_df['dg1/F1(stat)'].iloc[i] * F1_IpQE_value)
    G1syscalc.append(data_df['dg1/F1(sys)'].iloc[i] * F1_IpQE_value)

data_df['g1'] = G1calc
data_df['dg1(stat)'] = G1errcalc
data_df['dg1(sys)'] = G1syscalc
data_df['dg1(tot)'] = np.sqrt(data_df['dg1(stat)']**2 + data_df['dg1(sys)']**2)
data_df.round(4)

print(data_df['g1'].head())
# %%
data_df.to_csv(f'{dataset}.csv', index=False)
# %%
