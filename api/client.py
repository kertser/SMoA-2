import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import requests
from flask import Flask

# Инициализация Flask-сервера
server = Flask(__name__)

# Инициализация Dash-приложения
app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dashboard/',
    requests_pathname_prefix='/dashboard/'
)

# Определяем интерфейс приложения Dash
app.layout = html.Div([
    html.H1("Agent Graph Interaction"),
    dcc.Input(id="input-box", type="text", placeholder="Enter your query", debounce=True),
    html.Button('Submit', id='submit-button', n_clicks=0),
    cyto.Cytoscape(
        id='cytoscape-graph',
        layout={'name': 'breadthfirst'},
        style={'width': '100%', 'height': '500px'},
        elements=[]
    ),
    html.Div(id="output-response")
])

@app.callback(
    [Output('cytoscape-graph', 'elements'),
     Output('output-response', 'children')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')]
)
def update_graph(n_clicks, user_input):
    if n_clicks > 0 and user_input:
        try:
            response = requests.post("http://localhost:5000/api/message", json={'message': user_input})
            if response.status_code == 200:
                data = response.json()
                graph_data = data['response']['graph']
                final_response = data['response']['answer']
                return graph_data, f"Response: {final_response}"
        except Exception as e:
            return [], f"Error: {str(e)}"
    return [], "Enter a valid query."
