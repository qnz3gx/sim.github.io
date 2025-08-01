import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def import_csv_with_pandas(file_path):
    data = pd.read_csv(file_path)
    return data

file_path = '/Users/scarlettimorse/PycharmProjects/sim.github.io/ProtonData.csv'
P_df = import_csv_with_pandas(file_path)

columns_to_check = ['x', 'Q2']
plot_df = P_df.dropna(subset=columns_to_check)

plot_df['A1(x,Q2)'] = pd.to_numeric(plot_df['A1'], errors='coerce')
plot_df['G1F1(x,Q2)'] = pd.to_numeric(plot_df['g1/F1'], errors='coerce')

fig = go.Figure()

experiments = plot_df['Experiment'].unique()
g1f1exp = []
a1exp = []

for exp in experiments:
    df_exp = plot_df[plot_df['Experiment'] == exp]
    
    if not df_exp['G1F1(x,Q2)'].dropna().empty:
        g1f1exp.append(exp)
    
    if not df_exp['A1(x,Q2)'].dropna().empty:
        a1exp.append(exp)

annotations = []

symbol_map_g1f1 = {
    'CLAS_EG1': 'circle',
    'CLAS_EG1b': 'square',
    'SLAC_E143': 'star',
    'SLAC_E155': 'cross'
}

symbol_map_a1 = {
    'CLAS_EG1b': 'square-open',
    'SLAC_E143': 'star-open',
    'COMPASS_(JAM22)': 'triangle-up-open',
    'SMC': 'diamond-open'
}

color_map_g1f1= {
    'CLAS_EG1': 'firebrick',
    'CLAS_EG1b': 'slategray',
    'SLAC_E143': 'cadetblue',
    'SLAC_E155': 'purple',
    'HERMES': 'green'
}

color_map_A1= {
    'CLAS_EG1b': 'lavender',
    'SLAC_E143': 'lightblue',
    'COMPASS_(JAM22)': 'lightgray'
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

for exp in sorted(g1f1exp):
    exp_df = plot_df[plot_df['Experiment'] == exp]
    if exp_df['G1F1(x,Q2)'].dropna().empty:
        continue

    symbol1 = symbol_map_g1f1.get(exp, 'circle')
    color1 = color_map_g1f1.get(exp,'black')
    trace = go.Scatter(
        x=exp_df['x'],
        y=exp_df['G1F1(x,Q2)'],
        mode='markers',
        name=f"{str(exp)} ",
        error_y=dict(
        type='data',
        array=exp_df['dg1/F1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol1, color = color1),
        legendgroup=f"{exp}_g1f1",
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
    exp2_df = plot_df[plot_df['Experiment'] == exp2]
    if exp2_df['A1(x,Q2)'].dropna().empty:
        continue

    symbol2 = symbol_map_a1.get(exp2, 'circle')
    color2 = color_map_A1.get(exp2,'black')
    trace = go.Scatter(
        x=exp2_df['x'],
        y=exp2_df['A1(x,Q2)'],
        mode='markers',
        name=str(exp2),
        error_y=dict(
        type='data',
        array=exp2_df['dA1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol2, color = color2),
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
    title='g\u2081<sup>p</sup>/F\u2081<sup>p</sup>(x,Q²) and A\u2081<sup>p</sup>(x,Q²) vs X',
    xaxis_title='X',
    yaxis_title='g\u2081<sup>p</sup>/F\u2081<sup>p</sup>, A\u2081<sup>p</sup>',
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
                        "marker.color": ['gray'],
                        "line.color": ['gray']
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
            y=0.67,
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

fig.write_html("g1F1,A1(p)_vs_X.html")

pio.write_html(
    fig,
    file='g1F1,A1(p)_vs_X.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'g1F1,A1(p)_vs_X.html',
            'scale': 2
        }
    }
)