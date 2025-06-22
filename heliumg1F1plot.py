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

def xW(Q2):
    xW = Q2/(Q2 + 4 - 938.272/((3*10**8)**2))
    return xW

# wtwo = pd.DataFrame()
# wtwo['X'] = xW(h_df['X'])
# #w_df = xbins(wtwo)
# g1F1 = h_df.groupby('X_index')['g1/F1(x,Q2)'].mean()
# w_df['g1/F1'] = wtwo['X_index'].map(g1s)
# w_df.dropna(subset='g1/F1', inplace=True)

fig = go.Figure()

experiments = h_df['Experiment'].unique()
annotations = []

symbol_map = {
    'Zheng': 'circle',
    'Solvignon': 'triangle-up',
    'Flay': 'square'
}

open_symbol_map = {
    'Zheng': 'circle-open',
    'Solvignon': 'triangle-up-open',
    'Flay': 'square-open'
}

colors = {
    'Zheng': 'cornflowerblue',
    'Solvignon': 'blueviolet',
    'Flay': 'goldenrod'
}

for exp in experiments:
    exp_df = h_df[h_df['Experiment'] == exp]
    symbol = symbol_map.get(exp, 'circle')
    color = colors.get(exp, 'black')
    fig.add_trace(go.Scatter(
        x=exp_df['X'],
        y=exp_df['g1/F1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['dg1/F1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol, color=color),
        legendgroup=str(exp),
        showlegend=True
    ))
    
    symbol2 = open_symbol_map.get(exp, 'circle')
    fig.add_trace(go.Scatter(
        x=exp_df['X'],
        y=exp_df['A1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['dA1(tot)'],
        visible=True,
        thickness=1
    ),
        marker=dict(size=6, symbol=symbol2, color=color),
        legendgroup=str(exp),
        showlegend=False
    ))

    # xdata = w_df['Q2'].values
    # ydata = w_df['g1/F1'].values
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
    #     x=x_line[0],
    #     y=y_line[0],
    #     text=f"W = 2GeV",
    #     showarrow=False,
    #     xshift=0,
    #     yshift=10,
    #     font=dict(size=10, color="black"),
    #     ))

fig.update_layout(
    title='g\u2081<sup><sup>3</sup>He</sup>/F\u2081(x,Q²) vs Q²',
    xaxis_title='Q² (GeV²)',
    yaxis_title='g\u2081<sup><sup>3</sup>He</sup>/F\u2081 + 2.6 - 0.15i',
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
        )
    ]
)

fig.write_html("g1F1(3He)_vs_Q2.html")

pio.write_html(
    fig,
    file='g1F1(3He)_vs_Q2.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'g1F1(3He)_vs_Q2_plot',
            'height': 600,
            'width': 800,
            'scale': 2
        }
    }
)