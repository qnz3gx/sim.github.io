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
plot_df['A1(x,Q2)'] = pd.to_numeric(plot_df['A1'], errors='coerce') + 2 - 0.12 * pd.to_numeric(plot_df['X_index'], errors='coerce')
plot_df['G1F1(x,Q2)'] = pd.to_numeric(plot_df['g1/F1'], errors='coerce') + 2 - 0.12 * pd.to_numeric(plot_df['X_index'], errors='coerce')

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
    'Kramer': 'diamond',
    'HERMES': 'pentagon',
    'SMC': 'hexagon-open',
    'SLAC_E143': 'star-open',
    'SLAC_E155': 'cross-open',
    'COMPASS': 'triangle-up-open'
}

for exp in experiments:
    exp_df = plot_df[plot_df['Experiment'] == exp]
    symbol = symbol_map.get(exp, 'circle')
    fig.add_trace(go.Scatter(
        x=exp_df['Q2'],
        y=exp_df['G1F1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['dg1/F1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol),
        legendgroup=str(exp),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=exp_df['Q2'],
        y=exp_df['A1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['dA1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol),
        legendgroup=str(exp),
        showlegend=False
    ))

    xdata = w_df['Q2'].values
    ydata = w_df['g1/F1'].values

    mask = np.isfinite(xdata) & np.isfinite(ydata)
    xdata = xdata[mask]
    ydata = ydata[mask]

    def exponential(x, a, b, c):
        return a*np.exp(-b*x) + c

    try:
        popt, _ = curve_fit(exponential, xdata, ydata, bounds=(0, np.inf), maxfev=10000)
        x_line = np.linspace(xdata.min(), xdata.max(), 229)
        y_line = exponential(x_line, *popt)

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='red', width=1, dash='solid'),
            name="W = 2 GeV",
            showlegend=False
        ))
    except RuntimeError:
        print("Fit failed for reciprocal function")

    annotations.append(dict(
        x=-0.240,
        y=1,
        text=f"W = 2 GeV",
        showarrow=False,
        xshift=0,
        yshift=0,
        font=dict(size=10, color="black"),
        ))
    
rightmost_by_bin = plot_df.loc[plot_df.groupby('X_index')['Q2'].idxmax()]
top_point = rightmost_by_bin.loc[rightmost_by_bin['G1F1(x,Q2)'].idxmax()]
annotations.append(dict(
    x=np.log10(top_point['Q2']),
    y=top_point['G1F1(x,Q2)'],
    text=f"(i={int(top_point['X_index'])})",
    showarrow=False,
    xshift=60,
    yshift=0,
    font=dict(size=10, color="black", family='Arial Black'),
))

bin_df = plot_df[plot_df['X_index'] == 10]
if not bin_df.empty:
    bin_point = bin_df.loc[bin_df['Q2'].idxmax()]
    annotations.append(dict(
        x=np.log10(bin_point['Q2']),
        y=bin_point['G1F1(x,Q2)'],
        text=f"(i=10)",
        showarrow=False,
        xshift=40,
        yshift=0,
        font=dict(size=10, color="black", family='Arial Black'),
    ))

fig.update_layout(
    title='g\u2081<sup>n</sup>/F\u2081(x,Q²) vs Q²',
    xaxis_title='log(Q²)',
    yaxis_title='g\u2081<sup>n</sup>/F\u2081(x,Q²) + 5.2 - 0.3i',
    template='plotly_white',
    annotations=annotations,
    xaxis=dict(type='log'),
    yaxis=dict(range=[-0.7, 2.5]),
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

fig.write_html("g1F1(n)_vs_Q2.html")

pio.write_html(
    fig,
    file='g1F1(n)_vs_Q2.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'g1F1(n)_vs_Q2_plot',
            'height': 600,
            'width': 800,
            'scale': 2
        }
    }
)