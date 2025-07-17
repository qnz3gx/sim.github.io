import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

jamp = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/proton_JAM22.csv")
jamd = pd.read_csv("/Users/scarlettimorse/PycharmProjects/sim.github.io/deuteron_JAM22.csv")
cjp = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/proton_CJ15nlo.csv")
cjd = pd.read_csv("/Users/scarlettimorse/PycharmProjects/PDFs/deuteron.csv")
prot = pd.read_csv("CompassProton.csv")
deut = pd.read_csv("CompassDeuteron.csv")

fig = go.Figure()

#og proton
fig.add_trace(go.Scatter(
    x=prot['X'],
    y=prot['A1'],
    mode='markers',
    marker=dict(size=6),
    name='A<sub>1</sub><sup>p</sup>',
    legendgroup='A1p',
    showlegend=True
))

#og deuteron
fig.add_trace(go.Scatter(
    x=deut['X'],
    y=deut['A1'],
    mode='markers',
    marker=dict(size=6),
    name='A<sub>1</sub><sup>d</sup>',
    legendgroup='A1d',
    showlegend=True
))

#Jam proton
fig.add_trace(go.Scatter(
    x=jamp['x'],
    y=prot['G1']/jamp['F1'],
    mode='markers',
    marker=dict(size=6),
    name='g<sub>1</sub><sup>p</sup>/F<sub>1</sub><sup>p</sup> (JAM22)',
    legendgroup='g1p/F1p',
    showlegend=True
))

#Jam deuteron
fig.add_trace(go.Scatter(
    x=jamd['x'],
    y=deut['G1']/jamd['F1'],
    mode='markers',
    marker=dict(size=6),
    name='g<sub>1</sub><sup>d</sup>/F<sub>1</sub><sup>d</sup> (JAM22)',
    legendgroup='g1d/F1d',
    showlegend=True
))

#PDF proton
fig.add_trace(go.Scatter(
    x=jamp['x'],
    y=prot['G1']/cjp['F1'],
    mode='markers',
    marker=dict(size=6),
    name='g<sub>1</sub><sup>p</sup>/F<sub>1</sub><sup>p</sup> (PDFs)',
    legendgroup='g1/F1',
    showlegend=True
))

#PDF deuteron
fig.add_trace(go.Scatter(
    x=jamp['x'],
    y=2*deut['G1']/cjd['F1'],
    mode='markers',
    marker=dict(size=6),
    name='g<sub>1</sub><sup>d</sup>/F<sub>1</sub><sup>d</sup> (PDFs)',
    legendgroup='g1/F1pdf',
    showlegend=True
))

fig.update_layout(
    title='g<sub>1</sub>/F<sub>1</sub> Compared to A<sub>1</sub>',
    xaxis_title='X',
    yaxis_title='g₁ᵖ/F₁ᵖ, A₁ᵖ',
    template='plotly_white'
)

pio.write_html(
    fig,
    file='solong.html',
    auto_open=True,
    config={
        'toImageButtonOptions': {
            'filename': 'solong',
            'scale': 2
        }
    }
)