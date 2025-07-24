import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotstyle_base as ps

traces = 6

ctandcj = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/neutron_COMPASS.csv")
cj = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/neutron_CJ15nlo.csv")
ct = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/neutron_CT18NNLO.csv")
oldcj = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/ogPDg1F1_CJ15nlo.csv")
oldjam = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/ogPDg1F1_JAM22.csv")
oldct = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/ogPDg1F1_CT18NNLO.csv")
jam = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/neutron_JAM22.csv")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ps.offset(ctandcj['x'], 0, traces),
    y=ctandcj['g1/F1'],
    mode='markers',
    marker=dict(size=6, symbol='circle'),
    error_y=dict(
        type='data',
        array=ctandcj['dg1/F1(tot)'],
        visible=True,
        thickness=0.5
    ),
    name='CJ15 and CT18',
    legendgroup='cjct',
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=ps.offset(cj['x'], 0, traces),
    y=cj['g1/F1'],
    mode='markers',
    marker=dict(size=6, symbol='star'),
    error_y=dict(
        type='data',
        array=np.sqrt(cj['dg1/F1(stat)'].values**2 + cj['dg1/F1(sys)'].values**2).round(4),
        visible=True,
        thickness=0.5
    ),
    name='CJ15nlo (recalculated)',
    legendgroup='cj',
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=ps.offset(ct['x'],1,traces),
    y=ct['g1/F1'],
    mode='markers',
    marker=dict(size=6, symbol='triangle-up'),
    error_y=dict(
        type='data',
        array=np.sqrt(ct['dg1/F1(stat)'].values**2 + ct['dg1/F1(sys)'].values**2).round(4),
        visible=True,
        thickness=0.5
    ),
    name='CT18NNLO (recalculated)',
    legendgroup='ct',
    showlegend=True
))

# fig.add_trace(go.Scatter(
#     x=ps.offset(oldcj['x'],2,traces),
#     y=oldcj['g1/F1n'],
#     mode='markers',
#     marker=dict(size=6, symbol='square-open'),
#     error_y=dict(
#         type='data',
#         array=np.sqrt(oldcj['dg1/F1n(stat)'].values**2 + oldcj['dg1/F1n(sys)'].values**2).round(4),
#         visible=True,
#         thickness=0.5
#     ),
#     name='CJ15nlo (original COMPASS)',
#     legendgroup='CJ',
#     showlegend=True
# ))

# fig.add_trace(go.Scatter(
#     x=ps.offset(oldct['x'],3,traces),
#     y=oldct['g1/F1n'],
#     mode='markers',
#     marker=dict(size=6, symbol='square'),
#     error_y=dict(
#         type='data',
#         array=np.sqrt(oldct['dg1/F1n(stat)'].values**2 + oldct['dg1/F1n(sys)'].values**2).round(4),
#         visible=True,
#         thickness=0.5
#     ),
#     name='CT18NNLO (original COMPASS)',
#     legendgroup='CT',
#     showlegend=True
# ))

fig.add_trace(go.Scatter(
    x=ps.offset(jam['x'],4,traces),
    y=jam['g1/F1'],
    mode='markers',
    marker=dict(size=6, symbol='diamond'),
    error_y=dict(
        type='data',
        array=np.sqrt(jam['dg1/F1(stat)']**2+jam['dg1/F1(sys)']**2).round(4),
        visible=True,
        thickness=0.5
    ),
    name='JAM22 Evolved PDFs (recalculated)',
    legendgroup='jam',
    showlegend=True
))

# fig.add_trace(go.Scatter(
#     x=ps.offset(oldjam['x'],5,traces),
#     y=oldjam['g1/F1n'],
#     mode='markers',
#     marker=dict(size=6, symbol='circle'),
#     error_y=dict(
#         type='data',
#         array=np.sqrt(oldjam['dg1/F1n(stat)'].values**2 + oldjam['dg1/F1n(sys)'].values**2).round(4),
#         visible=True,
#         thickness=0.5
#     ),
#     name='JAM22 (original COMPASS)',
#     legendgroup='oldjam',
#     showlegend=True
# ))

fig.update_layout(
    title="Comparison of COMPASS g₁<sup>n</sup>/F₁<sup>n</sup>",
    xaxis=dict(type="log", title="log(x)", title_font=dict(size=15), tickfont=dict(size=15)),
    yaxis=dict(title="g₁<sup>n</sup>/F₁<sup>n</sup>", title_font=dict(size=15), tickfont=dict(size=15)),
    legend=dict(
        x=0.05,
        y=0.05,
        xanchor='left',
        yanchor='bottom',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1.5
    ),
    template="plotly_white"
)

pio.write_html(
    fig,
    file='COMPASS_comparison.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'COMPASS_comparison.html',
            'scale': 2
        }
    }
)