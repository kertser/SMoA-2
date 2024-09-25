import dash
from dash import dcc, html
import dash_cytoscape as cyto

def create_graph_layout():
    """Creates a layout for the Dash app to visualize the agent interaction graph."""
    layout = html.Div([
        html.H1("Agent Interaction Graph"),
        dcc.Input(id="query-input", type="text", placeholder="Enter a query"),
        html.Button('Submit', id='submit-btn'),
        cyto.Cytoscape(
            id='agent-graph',
            layout={'name': 'breadthfirst'},
            style={'width': '100%', 'height': '500px'},
            elements=[]
        ),
        html.Div(id="graph-response")
    ])
    return layout
