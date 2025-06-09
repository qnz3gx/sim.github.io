import pandas as pd
import numpy as np
import plotly.express as px

def import_csv_with_pandas(file_path,r):
    data = pd.read_csv(file_path, skiprows=r)
    return data

file_path = '/Users/scarlettimorse/PycharmProjects/PythonProject/Website/Available_Polarized_Data.csv'
APD_df = import_csv_with_pandas(file_path,10)

experiment_mapping = {
    1: 'SLAC E142',
    2: 'SLAC E154',
    3: 'Zheng',
    4: 'Kramer',
    5: 'Flay',
    6: 'Solvignon',
    7: 'E97110',
    8: 'E94010',
}

APD_df['Experiment'] = APD_df['Experiment'].replace(experiment_mapping)
allData_df = APD_df.replace(-1000.0,'')
allData_df.to_csv("allData.csv", index=False)


experiment_groups = {
    experiment: df for experiment, df in APD_df.groupby('Experiment')
}

cleaned_experiment_groups = {}

for experiment, df in experiment_groups.items():
    cleaned_df = df.loc[:, ~(df == -1000.0).any()]
    cleaned_experiment_groups[experiment] = cleaned_df
    cleaned_experiment_groups[experiment].iloc[:, :-1].to_csv(f'{experiment}.csv', index=False)

# fig, axs = plt.subplots(8, 6, sharex=True, sharey=True)
# axs = axs.flatten()
# for i, q2 in enumerate(unique_q2):
#     ax = axs[i]
#     subset = polarized_3he_df[polarized_3he_df['Q2'] == q2]
#     ax.errorbar(subset['W'] != -1000.0, subset['G1.mes'] != -1000.0, yerr=subset['G1.mes.err'] != -1000.0, fmt='o-', label='G1')
#     ax.errorbar(subset['W'] != -1000.0, subset['G2.mes'] != -1000.0, yerr=subset['G2.mes.err'] != -1000.0, fmt='o-', label='G2')
# plt.tight_layout()
# plt.show()

# unique_q2_3 = sorted(cleaned_experiment_groups['Zheng']['Q2'].unique())
# fig, axs = plt.subplots(1, 3, figsize=(6, 4), sharex=True, sharey=True)
# axs = axs.flatten()

# for i, q2 in enumerate(unique_q2_3[:3]):
#     ax = axs[i]
#     subset = cleaned_experiment_groups['Zheng'][cleaned_experiment_groups['Zheng']['Q2'] == q2]
#     ax.errorbar(subset['W'], subset['G1.mes'], yerr=subset['G1.mes.err'], fmt='o-', label='G1', capsize=3, elinewidth=0.8)
#     ax.errorbar(subset['W'], subset['G2.mes'], yerr=subset['G2.mes.err'], fmt='o-', label='G2', capsize=3, elinewidth=0.8)
#     ax.errorbar(subset['W'], subset['G1F1.mes'], yerr=subset['G1F1.mes.err'], fmt='o-', label='G1F1', capsize=3, elinewidth=0.8)
#     ax.errorbar(subset['W'], subset['G2F1.mes'], yerr=subset['G2F1.mes.err'], fmt='o-', label='G2F1', capsize=3, elinewidth=0.8)
#     ax.set_title(f'Q² = {q2}')
#     ax.set_xlabel('W')
#     ax.set_ylabel('Measured Quantities')
#     ax.legend()

# fig.suptitle('Measured Quantities vs W for different Q² values (Zheng)')
# plt.tight_layout()
# plt.savefig("Zheng.png")
# plt.show()

# flay_df = cleaned_experiment_groups['Flay'].copy()
# flay_df.loc[:, 'Q2'] = pd.to_numeric(flay_df['Q2'], errors='coerce').round()
# cleaned_experiment_groups['Flay'] = flay_df
# mostly_unique = cleaned_experiment_groups['Flay']['Q2'].round()
# unique_q2_5 = sorted(mostly_unique.unique())

# fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
# axs = axs.flatten()

# for i, q2 in enumerate(unique_q2_5[:6]):
#     ax = axs[i]
#     subset = cleaned_experiment_groups['Flay'][cleaned_experiment_groups['Flay']['Q2'] == q2]
#     ax.scatter(subset['W'], subset['G1F1.mes'], label='G1F1')
#     ax.errorbar(subset['W'], subset['G1F1.mes'], yerr=subset['G1F1.mes.err'], capsize=3, elinewidth=0.8, linestyle='none')
#     ax.scatter(subset['W'], subset['G2F1.mes'], label='G2F1')
#     ax.errorbar(subset['W'], subset['G2F1.mes'], yerr=subset['G2F1.mes.err'], capsize=3, elinewidth=0.8, linestyle='none')
#     ax.set_title(f'Q² \u2248 {q2}')
#     ax.set_xlabel('W')
#     ax.set_ylabel('Measured Quantities')
#     ax.legend()

# fig.suptitle('Measured Quantities vs W for different Q² values (Flay)')
# plt.tight_layout()
# plt.savefig("Flay.png")
# plt.show()