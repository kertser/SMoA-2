import logging
import os
from flask import Flask, request, jsonify
from graph.agent_graph import AgentGraph  # Исправленный импорт
from api.utility import handle_openai_error, handle_internal_error, handle_request_error  # Исправленный импорт
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/secrets.env")

# Flask app initialization
app = Flask(__name__)

# Global agent graph instance
agent_graph = AgentGraph()


@app.route('/')
def index():
    """Simple index page."""
    return jsonify({"message": "Welcome to the Agent API. Use /api/message to interact."})


@app.route('/status', methods=['GET'])
def status():
    """Check if the server is ready."""
    return jsonify({'status': 'ready'}), 200


@app.route('/api/message', methods=['POST'])
async def handle_message():
    """Handle incoming messages from the client."""
    data = request.get_json()
    if not data or 'message' not in data:
        return handle_request_error("No message provided")

    user_message = data['message']
    try:
        final_response = await agent_graph.run(user_message)
        return jsonify({'response': final_response})
    except Exception as e:
        return handle_internal_error(e)


def create_app():
    """Create Flask app."""
    return app
