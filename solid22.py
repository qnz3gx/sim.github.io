import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def import_csv_with_pandas(file_path):
    data = pd.read_csv(file_path)
    return data

file_path = '/Users/scarlettimorse/PycharmProjects/sim.github.io/dA1nvals.csv'
ND_df = import_csv_with_pandas(file_path)

columns_to_check = ['x', 'Q2', 'dA1n']
h_df = ND_df.dropna(subset=columns_to_check)
print(min(h_df['Q2']),max(h_df['Q2']))

centers = np.array([0.0001,0.0002,0.0003,0.0005,0.0008,0.0013,0.0020,0.0032,0.0051,0.0082,0.0129,0.0205,0.0325,0.0515,0.0815,0.1292,0.2048,0.3246,0.5145,0.8155])
#centers = np.array([0.02, 0.08, 0.14, 0.18, 0.26, 0.3, 0.34, 0.36, 0.42, 0.48, 0.56, 0.6, 0.62, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84])

def xbins(set):
    midpoints = (centers[:-1] + centers[1:]) / 2
    min_x = set['x'].min()
    max_x = set['x'].max()

    edges = np.concatenate([
        [min(min_x, centers[0] - (centers[1] - centers[0]) / 2)],
        midpoints,
        [max(max_x, centers[-1] + (centers[-1] - centers[-2]) / 2)]
    ])

    set = set.copy()
    set.loc[:, 'X_index'] = pd.cut(set['x'], bins=edges, labels=False, include_lowest=True)
    return set

plot_df = xbins(h_df)
plot_df['dA1(x,Q2)'] = 0
for i in range(len(centers)):
    plot_df.loc[plot_df['X_index']==i,'dA1(x,Q2)'] = plot_df.loc[plot_df['X_index']==i,'dA1n'] - 3 * np.log10(centers[i])

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

fig = go.Figure()

annotations = []
colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
    "#f7b6d2",  # light pink
    "#c7c7c7",  # light gray
    "#dbdb8d",  # light yellow-green
    "#9edae5"   # light cyan
]

fig.add_trace(go.Scatter(
    x=plot_df['Q2'],
    y=plot_df['dA1(x,Q2)'],
    mode='markers',
    name="dA<sub>1</sub><sup>n</sup>",
    marker=dict(size=0, color='rgba(0,0,0,0)'),
    legendgroup="dA<sub>1</sub>",
    showlegend=False
))

for i in range(len(centers)):
    fig.add_trace(go.Scatter(
        x=plot_df.loc[plot_df['X_index'] == i, 'Q2'],
        y=plot_df.loc[plot_df['X_index'] == i, 'dA1(x,Q2)'],
        mode='markers',
        marker=dict(size=4, color=colors[i], opacity=0.6),
        error_y=dict(
            type='data',
            array=plot_df.loc[plot_df['X_index'] == i, 'dA1n'],
            visible=True,
            thickness=1
        ),
        name=f'i={i}',
        showlegend=False
    ))
    
rightmost_by_bin = plot_df.loc[plot_df.groupby('X_index')['Q2'].idxmax()]
top_point = rightmost_by_bin.loc[rightmost_by_bin['dA1(x,Q2)'].idxmax()]
n=0
for _, row in rightmost_by_bin.iterrows():

    center = centers[int(row['X_index'])]

    if center.round(4) == 0.5145:
        annotations.append(dict(
        x=np.log10(18.7),
        y=0.8814,
        text=f"x={center:.4f}",
        showarrow=False,
        font=dict(size=9, color=colors[n]),
        xshift=45
    ))
    elif center.round(4) == 0.8155:
        annotations.append(dict(
        x=np.log10(22.3),
        y=0.4288,
        text=f"x={center:.4f}",
        showarrow=False,
        font=dict(size=9, color=colors[n]),
        xshift=45
    ))
    else:
        annotations.append(dict(
            x=np.log10(row['Q2']),
            y=row['dA1(x,Q2)'],
            text=f"x={center:.4f}",
            showarrow=False,
            font=dict(size=9, color=colors[n]),
            xshift=45
        ))
    n = n+1

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
    # layer="below"
)

major_ticks = [0.1, 1, 10, 100, 1000, 10000, 100000]
major_labels = [r"10⁻¹", r"10⁰", r"10¹", r"10²", r"10³", r"10⁴", r"10⁵"]

minor_ticks = []
for start in [0.1, 1, 10, 100, 1000, 10000]:
    minor_ticks.extend([start * i for i in range(2, 10)])

tickvals = major_ticks + minor_ticks
ticktext = major_labels + [""] * len(minor_ticks)

fig.update_layout(
    title='dA\u2081<sup>n</sup>(x,Q²) vs Q²',
    xaxis=dict(type='log',
               tickvals=tickvals,
               ticktext=ticktext,
               ticks='outside',
               ticklen=6,
               tickwidth=1,
               tickfont=dict(size=12),
               title='Q² (GeV²)',
               range=[-1,5]
    ),
    yaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=2,
        ticks='outside',
        ticklen=6,
        tickwidth=1,
        tickfont=dict(size=12),
        title='dA₁ⁿ - 3log(x)',
        range=[0,10]
    ),
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
            pad={"r": 10, "t": 10},
        )
    ]
)

pio.write_html(
    fig,
    file='A1_vs_Q2.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'A1_vs_Q2_plot',
            'height': 800,
            'width': 600,
            'scale': 2
        }
    }
)