import pandas as pd
import numpy as np

def import_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def W(data):
    data_path = f"/Users/scarlettimorse/PycharmProjects/sim.github.io/{data}.csv"
    data_df = import_csv(data_path)
    M = 938.272/((3*10**8)**2)

    data_df['W'] = np.sqrt(data_df['Q2'] * (1 / data_df['X'] - 1) + M**2)
    data_df.to_csv(f"{data}.csv")

W('neutron_Kramer')
W('neutron_SLAC_E142')
W('neutron_SLAC_E154')