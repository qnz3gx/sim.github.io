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

file_path = '/Users/scarlettimorse/PycharmProjects/sim.github.io/NeutronData.csv'
ND_df = import_csv_with_pandas(file_path)

columns_to_check = ['x', 'Q2']
h_df = ND_df.dropna(subset=columns_to_check)

centers = np.array([0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7])

def xbins(set):
    midpoints = (centers[:-1] + centers[1:]) / 2
    min_x = set['x'].min()
    max_x = set['x'].max()

    edges = np.concatenate([
        [min(min_x, centers[0] - (centers[1] - centers[0]) / 2)],
        midpoints,
        [max(max_x, centers[-1] + (centers[-1] - centers[-2]) / 2)]
    ])

    bin_labels = np.arange(len(centers))

    set = set.copy()
    set.loc[:, 'X_index'] = pd.cut(set['x'], bins=edges, labels=False, include_lowest=True)
    return set

plot_df = xbins(h_df)
plot_df['A1(x,Q2)'] = pd.to_numeric(plot_df['A1'], errors='coerce')
plot_df['G1F1(x,Q2)'] = pd.to_numeric(plot_df['g1/F1'], errors='coerce')

def xW(Q2):
    xW = Q2/(Q2 + 4 - 938.272/((3*10**8)**2))
    return xW

wtwo = pd.DataFrame()
wtwo['Q2'] = plot_df['Q2']
wtwo['x'] = xW(plot_df['Q2'])
w_df = xbins(wtwo)
g1s = plot_df.groupby('X_index')['G1F1(x,Q2)'].mean()
w_df['g1/F1'] = w_df['X_index'].map(g1s)

fig = go.Figure()

experiments = plot_df['Experiment'].unique()
bins = sorted(plot_df['X_index'].unique())
annotations = []

symbol_map = {
    'SLAC_E154': 'circle',
    'SLAC_E142': 'square',
    'Zheng': 'hourglass',
    'HERMES': 'pentagon',
    'SLAC_E143': 'star-open',
    'SLAC_E155': 'cross-open',
    'COMPASS': 'triangle-up-open'
}

color_map_g1f1= {
    'SLAC_E154': 'brown',
    'SLAC_E142': 'green',
    'Zheng': 'purple',
    'HERMES': 'pink',
    'Flay': 'gold',
    'SLAC_E143': 'blue',
    'SLAC_E155': 'orange',
    'COMPASS': 'darkred'
}

color_map_A1= {
    'SLAC_E154': 'sienna',
    'SLAC_E142': 'lightgreen',
    'Zheng': 'magenta',
    'HERMES': 'fuchsia',
    'Flay': 'yellow',
    'SLAC_E143': 'lightblue',
    'SLAC_E155': 'peachpuff',
    'COMPASS': 'firebrick'
}

for exp in experiments:
    exp_df = plot_df[plot_df['Experiment'] == exp]
    if exp_df['G1F1(x,Q2)'].dropna().empty and exp_df['A1(x,Q2)'].dropna().empty:
        continue

    exp_df = plot_df[plot_df['Experiment'] == exp]
    symbol = symbol_map.get(exp, 'circle')
    color1 = color_map_g1f1.get(exp,'black')
    color2 = color_map_A1.get(exp,'black')
    fig.add_trace(go.Scatter(
        x=exp_df['x'],
        y=exp_df['G1F1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['dg1/F1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol, color = color1),
        legendgroup=str(exp),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=exp_df['x'],
        y=exp_df['A1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['dA1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol, color = color2),
        legendgroup=str(exp),
        showlegend=False
    ))

    xdata = w_df['x'].values
    ydata = w_df['g1/F1'].values

    mask = np.isfinite(xdata) & np.isfinite(ydata)
    xdata = xdata[mask]
    ydata = ydata[mask]

    # def exponential(x, a, b, c):
    #     return a*np.exp(-b*x) + c

    # try:
    #     popt, _ = curve_fit(exponential, xdata, ydata, bounds=(0, np.inf), maxfev=10000)
    #     x_line = np.linspace(xdata.min(), xdata.max(), 229)
    #     y_line = exponential(x_line, *popt)

    #     fig.add_trace(go.Scatter(
    #         x=x_line,
    #         y=y_line,
    #         mode='lines',
    #         line=dict(color='red', width=1, dash='solid'),
    #         name="W = 2 GeV",
    #         showlegend=False
    #     ))
    # except RuntimeError:
    #     print("Fit failed for reciprocal function")

    # annotations.append(dict(
    #     x=-0.240,
    #     y=1,
    #     text=f"W = 2 GeV",
    #     showarrow=False,
    #     xshift=0,
    #     yshift=0,
    #     font=dict(size=10, color="black"),
    #     ))

fig.update_layout(
    title='g\u2081<sup>n</sup>/F\u2081(x,Q²) vs X',
    xaxis_title='log(X)',
    yaxis_title='g\u2081<sup>n</sup>/F\u2081(x,Q²)',
    template='plotly_white',
    annotations=annotations,
    xaxis=dict(type='log',range=[-1.8,0.1]),
    yaxis=dict(range=[-1.2, 1.2]),
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
        )
    ]
)

fig.write_html("g1F1,A1(n)_vs_X.html")

pio.write_html(
    fig,
    file='g1F1,A1(n)_vs_X.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'g1F1,A1(n)_vs_X.html',
            'height': 600,
            'width': 800,
            'scale': 2
        }
    }
)