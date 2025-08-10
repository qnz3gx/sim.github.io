import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

compass = pd.read_csv('neutron_COMPASS.csv')

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=compass['x'],
    y=compass['g1'],
    mode='markers',
    name='dg1(stat)',
    error_y=dict(
        type='data',
        array=compass['dg1(stat)'],
        visible=True,
        thickness=1,
        color='blue'
    ),
    marker=dict(size=0.0001,opacity=0),
    legendgroup='dg1(stat)',
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name='dg1(stat)',
    marker=dict(size=6, color='blue'),
    legendgroup='dg1(stat)',
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=compass['x'],
    y=compass['dg1(model)'],
    fill='tozeroy',
    mode='lines',
    name='dg1(model)',
    marker=dict(size=0, color='purple'),
    legendgroup='dg1(model)',
    showlegend=True
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
    title='g\u2081<sup>n</sup> Uncertainty',
    xaxis_title='x',
    yaxis_title='g\u2081<sup>n</sup>',
    legend=dict(
        x=0.95,
        y=0.05,
        xanchor='right',
        yanchor='bottom',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1.5),
    template='plotly_white'
)


pio.write_html(
    fig,
    file='g1err.html',
    auto_open=True,
    config={
    'toImageButtonOptions': {
        'filename': 'g1err',
        'height': 600,
        'width': 800,
        'scale': 2
        }
    }
)
