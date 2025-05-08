import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

fs = 500
t = np.linspace(0, 1, fs, endpoint=False)

def my_custom_filter(signal, window_size=5):
    filtered = np.zeros_like(signal)
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2 + 1)
        filtered[i] = np.mean(signal[start:end])
    return filtered

app.layout = html.Div([
    html.H1("Signal Visualization with Custom Filter", style={'text-align': 'center'}),

    dcc.Graph(id='signal-graph'),

    html.Div([
        html.Label('Signal Type', style={'margin-bottom': '5px'}),
        dcc.Dropdown(
            id='signal-type-dropdown',
            options=[
                {'label': 'Sine', 'value': 'sin'},
                {'label': 'Square', 'value': 'square'},
                {'label': 'Sawtooth', 'value': 'sawtooth'}
            ],
            value='sin',
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

    html.Div([
        html.Div([
            html.Label('Amplitude'),
            dcc.Slider(id='amplitude-slider', min=0.1, max=2, step=0.1, value=1,
                       marks={i: str(i) for i in [0.1, 0.5, 1, 1.5, 2]})
        ], style={'margin-bottom': '20px'}),

        html.Div([
            html.Label('Frequency'),
            dcc.Slider(id='frequency-slider', min=1, max=100, step=1, value=5,
                       marks={i: str(i) for i in range(0, 101, 10)})
        ], style={'margin-bottom': '20px'}),

        html.Div([
            html.Label('Phase'),
            dcc.Slider(id='phase-slider', min=0, max=360, step=10, value=0,
                       marks={i: str(i) for i in range(0, 361, 90)})
        ], style={'margin-bottom': '20px'}),

        html.Div([
            html.Label('Noise Mean'),
            dcc.Slider(id='noise-mean-slider', min=-1, max=1, step=0.1, value=0,
                       marks={i: str(i) for i in [-1, -0.5, 0, 0.5, 1]})
        ], style={'margin-bottom': '20px'}),

        html.Div([
            html.Label('Noise Covariance'),
            dcc.Slider(id='noise-cov-slider', min=0, max=1, step=0.1, value=0.5,
                       marks={i: str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1]})
        ], style={'margin-bottom': '20px'}),

        html.Div([
            html.Label('Filter Window Size'),
            dcc.Slider(id='filter-slider', min=1, max=50, step=1, value=5,
                       marks={i: str(i) for i in [1, 10, 20, 30, 40, 50]})
        ], style={'margin-bottom': '20px'}),
    ], style={'width': '80%', 'margin': '0 auto'}),

    html.Div([
        html.Button('Reset', id='reset-btn', n_clicks=0,
                    style={'margin-right': '10px', 'padding': '5px 15px'}),
        html.Button('Toggle Noise', id='toggle-noise-btn', n_clicks=0,
                    style={'padding': '5px 15px'})
    ], style={'text-align': 'center', 'margin': '20px 0'}),

    html.Div(id='slider-output', style={'text-align': 'center', 'font-weight': 'bold'})
])

@app.callback(
    Output('frequency-slider', 'value'),
    Output('amplitude-slider', 'value'),
    Output('phase-slider', 'value'),
    Output('noise-mean-slider', 'value'),
    Output('noise-cov-slider', 'value'),
    Output('filter-slider', 'value'),
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_sliders(n):
    return 5, 1, 0, 0, 0.5, 5

@app.callback(
    Output('slider-output', 'children'),
    Input('frequency-slider', 'value')
)
def update_slider_output(value):
    return f"Current Frequency: {value} Hz"

@app.callback(
    Output('signal-graph', 'figure'),
    Input('frequency-slider', 'value'),
    Input('amplitude-slider', 'value'),
    Input('phase-slider', 'value'),
    Input('noise-mean-slider', 'value'),
    Input('noise-cov-slider', 'value'),
    Input('filter-slider', 'value'),
    Input('toggle-noise-btn', 'n_clicks'),
    Input('signal-type-dropdown', 'value')
)
def update_graph(freq, amplitude, phase, noise_mean, noise_cov, window_size, noise_clicks, signal_type):
    show_noise = noise_clicks % 2 == 1
    phase_rad = np.deg2rad(phase)
    if signal_type == 'sin':
        clean = amplitude * np.sin(2 * np.pi * freq * t + phase_rad)
    elif signal_type == 'square':
        clean = amplitude * np.sign(np.sin(2 * np.pi * freq * t + phase_rad))
    elif signal_type == 'sawtooth':
        clean = amplitude * (2 * (t * freq - np.floor(0.5 + t * freq)))

    noise = np.random.normal(noise_mean, noise_cov, t.shape)
    noisy = clean + noise
    filtered = my_custom_filter(noisy, window_size=window_size)

    traces = [
        go.Scatter(x=t, y=clean, mode='lines', name='Clean Signal', line=dict(color='blue')),
    ]
    if show_noise:
        traces.append(go.Scatter(x=t, y=noisy, mode='lines', name='Noisy Signal', line=dict(color='orange')))
    traces.append(go.Scatter(x=t, y=filtered, mode='lines', name='Filtered Signal', line=dict(color='green')))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Signal with Optional Noise and Custom Filter',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        yaxis=dict(range=[-2, 2])
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
