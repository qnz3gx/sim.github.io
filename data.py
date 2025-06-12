import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import linregress

def import_csv_with_pandas(file_path,r):
    data = pd.read_csv(file_path, skiprows=r)
    return data

file_path = '/Users/scarlettimorse/PycharmProjects/PythonProject/Website/Available_Polarized_Data.csv'
APD_df = import_csv_with_pandas(file_path,10)

columns_to_check = ['X', 'Q2', 'G1.mes']
h_df = APD_df[~APD_df[columns_to_check].isin([-1000]).any(axis=1)]

APD_df = APD_df.replace(-1000,'')
APD_df.to_csv("allData.csv", index=False)

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

h_df['Experiment'] = h_df['Experiment'].replace(experiment_mapping)

centers = np.array([0.0036, 0.0045, 0.0055, 0.007, 0.009, 0.012, 0.017, 0.024,
                    0.035, 0.049, 0.077, 0.12, 0.17, 0.22, 0.29, 0.41, 0.57, 0.74])

midpoints = (centers[:-1] + centers[1:]) / 2
min_x = h_df['X'].min()
max_x = h_df['X'].max()

edges = np.concatenate([
    [min(min_x, centers[0] - (centers[1] - centers[0]) / 2)],
    midpoints,
    [max(max_x, centers[-1] + (centers[-1] - centers[-2]) / 2)]
])

bin_labels = np.arange(len(centers))

h_df = h_df.copy()
h_df.loc[:, 'X_index'] = pd.cut(h_df['X'], bins=edges, labels=False, include_lowest=True)

h_df = h_df.dropna(subset=['X_index'])
h_df.loc[:, 'X_index'] = h_df['X_index'].astype(int)

plot_df = h_df.copy()
plot_df['G1(x,Q2)'] = plot_df['G1.mes'] + 2.6 - 0.15 * plot_df['X_index']

fig = go.Figure()

experiments = plot_df['Experiment'].unique()
bins = sorted(plot_df['X_index'].unique())
annotations = []

for exp in experiments:
    exp_df = plot_df[plot_df['Experiment'] == exp]
    fig.add_trace(go.Scatter(
        x=exp_df['Q2'],
        y=exp_df['G1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['G1.mes.err'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6),
        legendgroup=str(exp),
        showlegend=True
    ))

for bin_idx in bins:
    bin_df = plot_df[plot_df['X_index'] == bin_idx].sort_values(by='Q2')
    
    if len(bin_df) > 1:
        slope, intercept, _, _, _ = linregress(bin_df['Q2'], bin_df['G1(x,Q2)'])
        
        line_x = bin_df['Q2']
        line_y = slope * line_x + intercept
        
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            line=dict(color='gray', width=1, dash='solid'),
            name=f'Best fit - X bin {bin_idx}',
            legendgroup=f'bin_{bin_idx}',
            showlegend=False
        ))

        x_label = line_x.iloc[-1]
        y_label = line_y.iloc[-1]
    
    elif len(bin_df) == 1:
        x_label = bin_df['Q2'].iloc[0]
        y_label = bin_df['G1(x,Q2)'].iloc[0]

    else:
        continue

    annotations.append(dict(
        x=x_label,
        y=y_label,
        text=f"x = {centers[bin_idx]:.4f}",
        showarrow=False,
        xshift=40,
        yshift=0,
        font=dict(size=10, color="black"),
        ))
    
top_point = plot_df.loc[plot_df['G1(x,Q2)'].idxmax()]
annotations.append(dict(
    x=top_point['Q2'],
    y=top_point['G1(x,Q2)'],
    text=f"(i={top_point['X_index']})",
    showarrow=False,
    xshift=90,
    yshift=0,
    font=dict(size=10, color="black", family='Arial Black'),
))

bin_12_df = plot_df[plot_df['X_index'] == 12]
if not bin_12_df.empty:
    bin_12_point = bin_12_df.loc[bin_12_df['Q2'].idxmax()]
    annotations.append(dict(
        x=bin_12_point['Q2'],
        y=bin_12_point['G1(x,Q2)'],
        text=f"(i=12)",
        showarrow=False,
        xshift=90,
        yshift=0,
        font=dict(size=10, color="black", family='Arial Black'),
    ))

fig.update_layout(
    title='g\u2081<sup><sup>3</sup>He</sup>(x,Q²) vs Q²',
    xaxis_title='Q² (GeV²)',
    yaxis_title='g\u2081<sup><sup>3</sup>He</sup>(x,Q²) + 2.6 - 0.15i',
    template='plotly_white',
    annotations=annotations
)

fig.write_html("g1(3He)_vs_Q2.html")

pio.write_html(
    fig,
    file='g1(3He)_vs_Q2.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'g1(3He)_vs_Q2_plot',
            'height': 600,
            'width': 800,
            'scale': 2
        }
    }
)