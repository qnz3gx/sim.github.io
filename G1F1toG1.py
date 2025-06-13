import pandas as pd
import numpy as np

#import csv file to pandas dataframe
def import_csv(file_path):
    data = pd.read_csv(file_path)
    return data

#csv paths
grid_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/XZ_table_3He_JAM_smeared_kpsv_onshell_ipol1_ipolres1_IA14_SF23_AC11_mod.csv"
G1_path = "/Users/scarlettimorse/PycharmProjects/sim.github.io/Flay.csv"

#dataframes
grid_df = import_csv(grid_path)
data_df = import_csv(G1_path)

G1calc = []
for i in range(len(data_df)):
    target_X = data_df['X'].iloc[i]
    target_Q2 = data_df['Q2'].iloc[i]
    
    # Compute distances from all points in grid_df
    distances = np.sqrt((grid_df['X'] - target_X)**2 + (grid_df['Q2'] - target_Q2)**2)
    
    # Find index of the nearest neighbor
    nearest_idx = distances.idxmin()
    
    # Get the corresponding F1_IpQE value
    F1_IpQE_value = grid_df.loc[nearest_idx, 'F1_IpQE']
    
    # Multiply with the G1F1.mes value
    G1calc.append(data_df['G1F1.mes'].iloc[i] * F1_IpQE_value)

data_df['G1.calc'] = G1calc
print(data_df.head())