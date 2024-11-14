import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import requests
import json
import networkx as nx
import plotly.graph_objects as go
import logging
import textwrap

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Agent System Interface", style={
        "textAlign": "center",
        "color": "#2c3e50",
        "marginBottom": "30px"
    }),
    dcc.Textarea(
        id="input-prompt",
        placeholder="Enter your prompt here...",
        style={
            "width": "100%",
            "height": 100,
            "padding": "10px",
            "borderRadius": "5px",
            "border": "1px solid #bdc3c7",
            "marginBottom": "20px"
        },
    ),
    html.Button(
        "Submit",
        id="submit-button",
        n_clicks=0,
        style={
            "backgroundColor": "#3498db",
            "color": "white",
            "padding": "10px 20px",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer",
            "marginBottom": "20px"
        }
    ),
    dcc.Graph(
        id="output-graph",
        style={"height": "800px", "marginBottom": "30px"}
    ),
    html.Div([
        html.H3("Summary", style={"color": "#2c3e50"}),
        html.Div(id="output-summary", style={
            "whiteSpace": "pre-line",
            "padding": "15px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "5px",
            "marginBottom": "20px"
        }),
        html.H3("Detailed Path", style={"color": "#2c3e50"}),
        html.Div(id="output-detailed", style={
            "whiteSpace": "pre-line",
            "padding": "15px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "5px",
            "marginBottom": "20px"
        }),
        html.H3("Metrics", style={"color": "#2c3e50"}),
        html.Div(id="output-metrics", style={
            "whiteSpace": "pre-line",
            "padding": "15px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "5px"
        })
    ], style={"margin": "0 20px"})
], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"})

def create_hierarchical_layout(G):
    """Create a hierarchical layout suitable for flowcharts from left to right."""
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args="-Grankdir=LR")
    except Exception as e:
        print("Graphviz layout failed, defaulting to spring layout:", e)
        pos = nx.spring_layout(G)  # Fallback if Graphviz is unavailable
    return pos

def get_node_style(node_type):
    """Define visual styles for different node types."""
    styles = {
        'start': {
            'symbol': 'circle',
            'size': 40,
            'color': '#2ecc71',
            'line_color': '#27ae60',
            'line_width': 2
        },
        'prompt': {
            'symbol': 'circle',
            'size': 35,
            'color': '#3498db',
            'line_color': '#2980b9',
            'line_width': 2
        },
        'task': {
            'symbol': 'square',
            'size': 35,
            'color': '#9b59b6',
            'line_color': '#8e44ad',
            'line_width': 2
        },
        'end': {
            'symbol': 'hexagon',
            'size': 40,
            'color': '#e74c3c',
            'line_color': '#c0392b',
            'line_width': 2
        }
    }
    return styles.get(node_type, styles['task'])

def wrap_text(text, width):
    """Wrap text to the specified width."""
    return '<br>'.join(textwrap.wrap(text, width=width))

@app.callback(
    [
        Output("output-graph", "figure"),
        Output("output-summary", "children"),
        Output("output-detailed", "children"),
        Output("output-metrics", "children"),
    ],
    Input("submit-button", "n_clicks"),
    State("input-prompt", "value")
)
def update_output(n_clicks, prompt):
    if not prompt:
        return {}, "Please enter a prompt.", "", ""

    try:
        response = requests.post("http://localhost:5000/api/message", json={"message": prompt})

        try:
            data = response.json()
        except ValueError:
            return {}, "Error: Received an invalid JSON response from the server.", "", ""

        # Extract data
        graph_data = data.get("graph_data", [])
        summary_answer = data.get("summary_answer", "No summary available.")
        detailed_path = data.get("detailed_path", "No details available.")
        quality_score = data.get("quality_score", "N/A")
        processing_time = data.get("processing_time", "N/A")

        metrics_output = f"Processing Time: {processing_time} ms | Quality Score: {quality_score}"

        # Create graph using NetworkX
        G = nx.DiGraph()

        # Add nodes and edges to the graph
        nodes_data = []
        edges_data = []
        for element in graph_data:
            if 'data' in element:
                if 'id' in element['data'] and 'label' in element['data']:
                    nodes_data.append(element)
                elif 'source' in element['data'] and 'target' in element['data']:
                    edges_data.append(element)

        # Add nodes first
        for node in nodes_data:
            node_id = node['data']['id']
            node_label = node['data']['label']
            node_type = node['data'].get('type', 'task')
            G.add_node(node_id, label=node_label, type=node_type)

        # Then add edges
        for edge in edges_data:
            source = edge['data']['source']
            target = edge['data']['target']
            edge_label = edge['data'].get('label', '')
            G.add_edge(source, target, label=edge_label)

        # Use hierarchical layout from left to right
        pos = create_hierarchical_layout(G)

        # Adjust positions for left-to-right flow
        # (No need to invert axes since 'rankdir=LR' handles it)

        # Create edge traces with labels
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_info = G.edges[edge].get('label', '')

            # Edge line
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(
                    width=1.5,
                    color='#95a5a6',
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

            # Edge label at the midpoint
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            edge_label_trace = go.Scatter(
                x=[mid_x],
                y=[mid_y],
                text=[wrap_text(edge_info, width=20)],
                mode='text',
                textfont=dict(size=10),
                textposition='top center',
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_label_trace)

        # Create node traces by type
        node_traces = []
        for node_type in set(nx.get_node_attributes(G, 'type').values()):
            nodes_of_type = [node for node, attr in G.nodes(data=True) if attr.get('type') == node_type]
            if not nodes_of_type:
                continue

            style = get_node_style(node_type)

            node_trace = go.Scatter(
                x=[pos[node][0] for node in nodes_of_type],
                y=[pos[node][1] for node in nodes_of_type],
                mode='markers+text',
                text=[wrap_text(G.nodes[node]['label'], width=20) for node in nodes_of_type],
                textposition="bottom center",
                textfont=dict(size=12),
                marker=dict(
                    symbol=style['symbol'],
                    size=style['size'],
                    color=style['color'],
                    line=dict(color=style['line_color'], width=style['line_width'])
                ),
                name=node_type.capitalize(),
                hoverinfo='text',
                showlegend=True
            )
            node_traces.append(node_trace)

        # Combine all traces
        all_traces = edge_traces + node_traces

        # Create the figure
        figure = {
            'data': all_traces,
            'layout': go.Layout(
                title='Task Flowchart',
                titlefont=dict(size=16),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=40, l=40, r=40, t=40),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                ),
                height=800,
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#2c3e50',
                font_size=12,
                autosize=True,
                legend=dict(
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                )
            )
        }

        return figure, summary_answer, detailed_path, metrics_output

    except requests.exceptions.RequestException as e:
        logging.error(f"Server error: {e}")
        return {}, f"Server error: Unable to reach the API. {str(e)}", "", ""

if __name__ == "__main__":
    app.run_server(debug=True)
