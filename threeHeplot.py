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

columns_to_check = ['X', 'Q2', 'g1']
h_df = APD_df[APD_df['Experiment'] != 'E97110']

centers = np.array([0.0036, 0.0045, 0.0055, 0.007, 0.009, 0.012, 0.017, 0.024,
                    0.035, 0.049, 0.077, 0.12, 0.17, 0.22, 0.29, 0.41, 0.57, 0.74])

def xbins(set):
    midpoints = (centers[:-1] + centers[1:]) / 2
    min_x = set['X'].min()
    max_x = set['X'].max()

    edges = np.concatenate([
        [min(min_x, centers[0] - (centers[1] - centers[0]) / 2)],
        midpoints,
        [max(max_x, centers[-1] + (centers[-1] - centers[-2]) / 2)]
    ])

    bin_labels = np.arange(len(centers))

    set = set.copy()
    set.loc[:, 'X_index'] = pd.cut(set['X'], bins=edges, labels=False, include_lowest=True)

    set = set.dropna(subset=['X_index'])
    set.loc[:, 'X_index'] = set['X_index'].astype(int)
    return set

plot_df = xbins(h_df)
plot_df['G1(x,Q2)'] = pd.to_numeric(plot_df['g1'], errors='coerce') + 2.6 - 0.15 * pd.to_numeric(plot_df['X_index'], errors='coerce')

def xW(Q2):
    xW = Q2/(Q2 + 4 - 938.272/((3*10**8)**2))
    return xW

wtwo = pd.DataFrame()
wtwo['Q2'] = plot_df['Q2']
wtwo['X'] = xW(plot_df['Q2'])
w_df = xbins(wtwo)
g1s = plot_df.groupby('X_index')['G1(x,Q2)'].mean()
w_df['g1'] = w_df['X_index'].map(g1s)

fig = go.Figure()

experiments = plot_df['Experiment'].unique()
bins = sorted(plot_df['X_index'].unique())
annotations = []

symbol_map = {
    'SLAC_E154': 'circle',
    'SLAC_E142': 'square',
    'Zheng': 'triangle-up',
    'Kramer': 'diamond',
    'Solvignon': 'star',
    'Flay': 'hourglass',
    'HERMES': 'pentagon'
}

plot_df = plot_df.dropna(subset=['X','Q2','G1(x,Q2)'])

for exp in experiments:
    exp_df = plot_df[plot_df['Experiment'] == exp]
    symbol = symbol_map.get(exp, 'circle')
    fig.add_trace(go.Scatter(
        x=exp_df['Q2'],
        y=exp_df['G1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['dg1(tot)'],
        visible=True,
        thickness=1,
        width=0
    ),
        marker=dict(size=6, symbol=symbol),
        legendgroup=str(exp),
        showlegend=True
    ))

    xdata = w_df['Q2'].values
    ydata = w_df['g1'].values
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
        x=0.57,
        y=y_line[0],
        text=f"W = 2 GeV",
        showarrow=False,
        xshift=0,
        yshift=10,
        font=dict(size=10, color="black"),
        ))

# for bin_idx in bins:
#     bin_idx=int(bin_idx)
#     bin_df = plot_df[plot_df['X_index'] == bin_idx].sort_values(by='Q2')
    
#     if len(bin_df) > 1:
#         slope, intercept, _, _, _ = linregress(bin_df['Q2'], bin_df['G1(x,Q2)'])
        
#         line_x = bin_df['Q2']
#         line_y = slope * line_x + intercept
        
#         fig.add_trace(go.Scatter(
#             x=line_x,
#             y=line_y,
#             mode='lines',
#             line=dict(color='gray', width=1, dash='solid'),
#             name=f'Best fit - X bin {bin_idx}',
#             legendgroup=f'bin_{bin_idx}',
#             showlegend=False
#         ))

#         x_label = line_x.iloc[-1]
#         y_label = line_y.iloc[-1]
    
#     elif len(bin_df) == 1:
#         x_label = bin_df['Q2'].iloc[0]
#         y_label = bin_df['G1(x,Q2)'].iloc[0]

#     else:
#         continue

#     annotations.append(dict(
#         x=x_label,
#         y=y_label,
#         text=f"x = {centers[bin_idx]:.4f}",
#         showarrow=False,
#         xshift=40,
#         yshift=0,
#         font=dict(size=10, color="black"),
#         ))
    
top_point = plot_df.loc[plot_df['G1(x,Q2)'].idxmax()]
annotations.append(dict(
    x=top_point['Q2'],
    y=top_point['G1(x,Q2)'],
    text=f"(i={int(top_point['X_index'])})",
    showarrow=False,
    xshift=16,
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
        xshift=22,
        yshift=0,
        font=dict(size=10, color="black", family='Arial Black'),
    ))

fig.add_shape(
    type="rect",
    xref="paper",
    yref="paper",
    x0=0,          
    y0=0,          
    x1=1,          
    y1=1,          
    line=dict(
        color="black",
        width=1,
    )
)

fig.update_layout(
    title='g\u2081<sup><sup>3</sup>He</sup>(x,Q²) vs Q²',
    xaxis_title='Q² (GeV²)',
    yaxis_title='g\u2081<sup><sup>3</sup>He</sup>(x,Q²) + 2.6 - 0.15i',
    template='plotly_white',
    annotations=annotations,
        legend=dict(
        xanchor="right",
        yanchor="top",
        x=0.99,
        y=0.99,
        bgcolor="white",
        bordercolor="black",
        font=dict(size=8),
        borderwidth=1
    )
    # updatemenus=[
    #     dict(
    #         type="buttons",
    #         direction="right",
    #         showactive=True,
    #         x=0.5,
    #         xanchor="center",
    #         y=1.1,
    #         yanchor="top",
    #         buttons=[
    #             dict(
    #                 label="Color",
    #                 method="update",
    #                 args=[{
    #                     "marker.color": [trace.marker.color if hasattr(trace.marker, "color") else 'gray' for trace in fig.data],
    #                     "line.color": [trace.line.color if hasattr(trace.line, "color") else 'gray' for trace in fig.data]
    #                 }],
    #             ),
    #             dict(
    #                 label="No Color",
    #                 method="update",
    #                 args=[{
    #                     "marker.color": ['gray' for trace in fig.data],
    #                     "line.color": ['gray' for trace in fig.data]
    #                 }],
    #             ),
    #         ],
    #         pad={"r": 10, "t": 10},
    #     )
    # ]
)

fig.write_html("g1(3He)_vs_Q2.html")

pio.write_html(
    fig,
    file='g1(3He)_vs_Q2.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'g1(3He)_vs_Q2_plot',
            'height': 800,
            'width': 600,
            'scale': 2
        }
    }
)