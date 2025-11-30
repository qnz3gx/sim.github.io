import pandas as pd

prot = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/ProtonData.csv")
new_prot = pd.DataFrame()
for exp in prot['Experiment'].unique():
    df = prot[prot['Experiment'] == exp]
    df_merged = (df.groupby(["x", "Q2"], as_index=False).agg(lambda col: col.dropna().iloc[0] if col.notna().any() else None))
    new_prot = pd.concat([new_prot, df_merged], axis=0, ignore_index=True)

print(len(new_prot), len(prot))
new_prot.to_csv("ProtonData.csv")