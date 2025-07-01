import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import linregress
from scipy.optimize import curve_fit

def import_csv_with_pandas(file_path,r):
    data = pd.read_csv(file_path, skiprows=r)
    return data

file_path = '/Users/scarlettimorse/PycharmProjects/sim.github.io/threeHedata.csv'
APD_df = import_csv_with_pandas(file_path,0)
h_df = APD_df[APD_df['Experiment'] != 'E97110']

columns_to_check = ['X']
h_df[columns_to_check] = h_df[columns_to_check].replace([np.inf, -np.inf], np.nan)
h_df.dropna(subset=columns_to_check, inplace=True)
h_df.dropna(subset=['A1', 'g1/F1'], how='all', inplace=True)

h_df['g1/F1(x,Q2)'] = pd.to_numeric(h_df['g1/F1'], errors='coerce')
h_df['A1(x,Q2)'] = pd.to_numeric(h_df['A1'], errors='coerce')

fig = go.Figure()

experiments = h_df['Experiment'].unique()
g1f1exp = []
a1exp = []

for exp in experiments:
    df_exp = h_df[h_df['Experiment'] == exp]

    if not df_exp['g1/F1(x,Q2)'].dropna().empty:
        g1f1exp.append(exp)
    
    if not df_exp['A1(x,Q2)'].dropna().empty:
        a1exp.append(exp)

annotations = []

symbol_map = {
    'SLAC E142': 'diamond',
    'Zheng': 'circle',
    'Solvignon': 'triangle-up',
    'Flay': 'square'
}

open_symbol_map = {
    'SLAC E142': 'diamond-open',
    'Zheng': 'circle-open',
    'Solvignon': 'triangle-up-open',
    'Flay': 'square-open'
}

colors = {
    'SLAC E142': 'coral',
    'Zheng': 'cornflowerblue',
    'Solvignon': 'blueviolet',
    'Flay': 'burlywood'
}

g1f1_trace_idxs = []
a1_trace_idxs = []
default_modes = []

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(size=0, opacity=0, color='rgba(0,0,0,0)'),
    name='                  ',
    legendgroup='g1f1title',
    showlegend=True
))
default_modes.append('markers')

for exp1 in sorted(g1f1exp):
    exp1_df = h_df[h_df['Experiment'] == exp1]
    if exp1_df['g1/F1(x,Q2)'].dropna().empty:
        continue

    symbol = symbol_map.get(exp1, 'circle')
    color = colors.get(exp1, 'black')
    trace = go.Scatter(
        x=exp1_df['X'],
        y=exp1_df['g1/F1(x,Q2)'],
        mode='markers',
        name=f"{str(exp1)}",
        error_y=dict(
        type='data',
        array=exp1_df['dg1/F1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol, color=color),
        legendgroup=f"{exp1}_g1f1",
        showlegend=True
    )
    fig.add_trace(trace)
    g1f1_trace_idxs.append(len(fig.data)-1)
    default_modes.append('markers')

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(size=0, opacity=0, color='rgba(0,0,0,0)'),
    name='           ',
    legendgroup='a1title',
    showlegend=True
))
default_modes.append('markers')

for exp2 in sorted(a1exp):
    exp2_df = h_df[h_df['Experiment'] == exp2]
    if exp2_df['A1(x,Q2)'].dropna().empty:
        continue
    
    symbol2 = open_symbol_map.get(exp2, 'circle')
    color=colors.get(exp2,'black')
    trace = go.Scatter(
        x=exp2_df['X'],
        y=exp2_df['A1(x,Q2)'],
        mode='markers',
        name=f"{str(exp2)}",
        error_y=dict(
        type='data',
        array=exp2_df['dA1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol2, color=color),
        legendgroup=f"{exp2}_a1",
        showlegend=True
    )
    fig.add_trace(trace)
    a1_trace_idxs.append(len(fig.data)-1)
    default_modes.append('markers')

error_y_values = [
    {"type": "data", "array": trace.error_y["array"], "thickness": 1} if "error_y" in trace else None
    for trace in fig.data
]

fig.update_layout(
    title='g\u2081<sup><sup>3</sup>He</sup>/F\u2081<sup><sup>3</sup>He</sup>(x,Q²) vs X',
    xaxis_title='x',
    yaxis_title='g\u2081<sup><sup>3</sup>He</sup>/F\u2081<sup><sup>3</sup>He</sup>',
    template='plotly_white',
    annotations=annotations,
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            showactive=True,
            x=0.5,
            xanchor="center",
            y=1.1,
            yanchor="top",
            buttons=[
                dict(
                    label="Color",
                    method="update",
                    args=[{
                        "marker.color": [trace.marker.color if hasattr(trace.marker, "color") else 'gray' for trace in fig.data],
                        "line.color": [trace.line.color if hasattr(trace.line, "color") else 'gray' for trace in fig.data]
                    }],
                ),
                dict(
                    label="No Color",
                    method="update",
                    args=[{
                        "marker.color": ['gray' for trace in fig.data],
                        "line.color": ['gray' for trace in fig.data]
                    }],
                ),
            ],
            pad={"r": 10, "t": 10},
        ),
        dict(
            type="buttons",
            direction="down",
            xanchor="left",
            x=1.04,
            y=1.00,
            yanchor="top",
            showactive=False,
            buttons=[
                dict(
                    label="— g₁/F₁ —",
                    method="update",
                    args=[{
                        "mode": default_modes,
                        "error_y": [{"visible": True, "array": error_y_values[i]["array"], "thickness": 1} for i in range(len(fig.data))]
                    }],
                    args2=[{
                        "mode": [
                            'none' if (trace.legendgroup and trace.legendgroup.endswith('_a1')) else default_modes[i]
                            for i, trace in enumerate(fig.data)
                        ],
                        "error_y": [
                            {"visible": False, "array": error_y_values[i]["array"], "thickness": 1} if (trace.legendgroup and trace.legendgroup.endswith('_a1')) else {"visible": True, "array": error_y_values[i]["array"], "thickness": 1}
                            for i, trace in enumerate(fig.data)
                        ]
                    }],
                ),
            ],
            pad={"r": 0, "t": 0},
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12, color="black")
        ),
        dict(
            type="buttons",
            direction="down",
            x=1.045,
            xanchor="left",
            y=0.84,
            yanchor="top",
            showactive=False,
            buttons=[
                dict(
                    label="— A₁ —",
                    method="update",
                    args=[{
                        "mode": default_modes,
                        "error_y": [{"visible": True, "array": error_y_values[i]["array"], "thickness": 1} for i in range(len(fig.data))]
                    }],
                    args2=[{
                        "mode": [
                            'none' if (trace.legendgroup and trace.legendgroup.endswith('_g1f1')) else default_modes[i]
                            for i, trace in enumerate(fig.data)
                        ],
                        "error_y": [
                            {"visible": False, "array": error_y_values[i]["array"], "thickness": 1} if (trace.legendgroup and trace.legendgroup.endswith('_g1f1')) else {"visible": True, "array": error_y_values[i]["array"], "thickness": 1}
                            for i, trace in enumerate(fig.data)
                        ]
                    }],
                ),
            ],
            pad={"r": 0, "t": 0},
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12, color="black")
        )
    ]
)

fig.write_html("g1F1(3He)_vs_X.html")

pio.write_html(
    fig,
    file='g1F1(3He)_vs_X.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'g1F1(3He)_vs_X_plot',
            'scale': 2
        }
    }
)