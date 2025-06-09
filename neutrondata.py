import pandas as pd
import numpy as np
import plotly.express as px

def import_csv_with_pandas(file_path):
    data = pd.read_csv(file_path)
    return data

file_path = '/Users/scarlettimorse/PycharmProjects/sim.github.io/NeutronData.csv'
ND_df = import_csv_with_pandas(file_path)

experiment_groups = {
    experiment: df for experiment, df in ND_df.groupby('Experiment')
}

for experiment, df in experiment_groups.items():
    experiment_groups[experiment].iloc[:, :-1].to_csv(f'neutron_{experiment}.csv', index=False)