import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

g1 = pd.read_csv('NeutronData.csv')

compass = g1[g1['Experiment'] == 'COMPASS']

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

fig.update_layout(
    title='g\u2081<sup>n</sup> Uncertainty',
    xaxis_title='x',
    yaxis_title='g\u2081<sup>n</sup>',
    template='plotly_white'
)

fig.write_html('g1err.html')
