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
PDF_df = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/g1p.csv")

columns_to_check = ['x', 'Q2', 'g1']
h_df = P_df.dropna(subset=columns_to_check)
h_df = h_df[h_df['Experiment'] != 'COMPASS_(CJ15+CT18)']

centers = np.array([0.0036, 0.0045, 0.0055, 0.007, 0.009, 0.012, 0.017, 0.024,
                    0.035, 0.049, 0.077, 0.12, 0.17, 0.22, 0.29, 0.41, 0.57, 0.74])

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
plot_df['G1(x,Q2)'] = plot_df['g1'] + 12.1 - 0.71 * plot_df['X_index']

small = []
large = []
j = 1
for j in range(len(centers)):
    if plot_df.loc[plot_df['X_index'] == j].empty:
        small.append(0)
        large.append(0)
    else:
        small.append(min(plot_df.loc[plot_df['X_index'] == j,'Q2']))
        large.append(max(plot_df.loc[plot_df['X_index'] == j,'Q2']))

def retrieve_g1(exs,grid_df):
    x_values = []
    Q2_values = []
    g1_values = []
    for target_X in exs:
        distance = np.abs(grid_df['x'] - target_X)
        x_match = distance.idxmin()
        matching_rows = grid_df[grid_df['x'] == grid_df.loc[x_match, 'x']]
        
        if not matching_rows.empty:
            x_values.extend([target_X] * len(matching_rows))
            Q2_values.extend(matching_rows['Q2'].values)
            g1_values.extend(matching_rows['g1'].values)

    return x_values, Q2_values, g1_values

g1PDF = pd.DataFrame()
g1PDF['x'],g1PDF['Q2'],g1PDF['g1'] = retrieve_g1(centers,PDF_df)
world_fit = xbins(g1PDF)
world_fit['G1(x,Q2)'] = world_fit['g1'] + 12.1 - 0.71 * world_fit['X_index']

def xW(Q2):
    xW = Q2/(Q2 + 4 - 938.272/((3*10**8)**2))
    return xW

wtwo = pd.DataFrame()
wtwo['Q2'] = plot_df['Q2']
wtwo['x'] = xW(plot_df['Q2'])
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
    'Zheng': 'hourglass',
    'Kramer': 'diamond',
    'HERMES': 'pentagon',
    'SMC': 'hexagon-open',
    'SLAC_E143': 'star-open',
    'SLAC_E155': 'cross-open',
    'COMPASS(JAM22)': 'triangle-up-open'#,
    # 'COMPASS_(CJ15+CT18)': 'triangle-up'
}

for exp in experiments:
    exp_df = plot_df[plot_df['Experiment'] == exp]
    symbol = symbol_map.get(exp, 'circle')
    fig.add_trace(go.Scatter(
        x= exp_df['Q2'],
        y=exp_df['G1(x,Q2)'],
        mode='markers',
        name=str(exp),
        error_y=dict(
        type='data',
        array=exp_df['dg1(tot)'],
        visible=True,
        thickness=1
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
        x_line = np.logspace(np.log10(xdata.min()), np.log10(xdata.max()), 229)
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
        x=-0.85,
        y=7.64,
        text=f"W = 2 GeV",
        showarrow=False,
        xshift=0,
        yshift=0,
        font=dict(size=10, color="black"),
        ))

for i in range(len(centers)):
    binned_df = world_fit[world_fit['x'] == centers[i]]
    if i > 9:
        domain = binned_df[(binned_df['Q2'] > small[i]) & (binned_df['Q2'] < large[i])]
    else:
        domain = binned_df[(binned_df['Q2'] > 0.045) & (binned_df['Q2'] < large[i])]
    fig.add_trace(go.Scatter(
        x=domain['Q2'],
        y=domain['G1(x,Q2)'],
        name = f'x = {centers[i]}',
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        showlegend=False
    ))

    try:
        y_val = world_fit.loc[
            (world_fit['x'] == centers[i]) & (world_fit['Q2'] == large[i]),
            'G1(x,Q2)'
        ].values[0]
    except IndexError:
        if not domain.empty:
            y_val = domain['G1(x,Q2)'].iloc[-1]
            x_val = domain['Q2'].iloc[-1]
        else:
            continue
    else:
        x_val = large[i]

    annotations.append(dict(
        x=np.log10(x_val)+0.15,
        y=y_val-0.1,
        text=f"x = {centers[i]}",
        showarrow=False,
        xshift=0,
        yshift=0,
        font=dict(size=10, color="black"),
    ))
    
rightmost_by_bin = plot_df.loc[plot_df.groupby('X_index')['Q2'].idxmax()]
top_point = rightmost_by_bin.loc[rightmost_by_bin['G1(x,Q2)'].idxmax()]
annotations.append(dict(
    x=np.log10(top_point['Q2'])-0.1,
    y=top_point['G1(x,Q2)'],
    text=f"(i={top_point['X_index']})",
    showarrow=False,
    xshift=80,
    yshift=0,
    font=dict(size=10, color="black", family='Arial Black'),
))

bin_df = plot_df[plot_df['X_index'] == 10]
if not bin_df.empty:
    bin_point = bin_df.loc[bin_df['Q2'].idxmax()]
    annotations.append(dict(
        x=np.log10(bin_point['Q2'])+0.1,
        y=bin_point['G1(x,Q2)'],
        text=f"(i=10)",
        showarrow=False,
        xshift=60,
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
    title='g\u2081<sup>p</sup>(x,Q²) vs Q²',
    xaxis_title='log(Q²)',
    yaxis_title='g\u2081<sup>p</sup>(x,Q²) + 12.1 - 0.71i',
    template='plotly_white',
    annotations=annotations,
    xaxis=dict(type='log'),
    yaxis=dict(range=[-2, 12.5]),
    legend=dict(
        yanchor='top',
        xanchor='right',
        x=0.99,
        y=0.99,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
        ),
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

fig.write_html("g1(p)_vs_Q2.html")

pio.write_html(
    fig,
    file='g1(p)_vs_Q2.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'g1(p)_vs_Q2_plot',
            'height': 800,
            'width': 600,
            'scale': 2
        }
    }
)