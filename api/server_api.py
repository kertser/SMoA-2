import asyncio
import logging
import time
from flask import Flask, request, jsonify
from graph.agent_graph import AgentGraph
from api.utility import handle_openai_error, handle_internal_error, handle_request_error
from dotenv import load_dotenv
from operators.reasoning import stop_classifier, simplicity_classifier

load_dotenv(dotenv_path="config/secrets.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

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
        # Start time for processing metrics
        start_time = time.time()

        # Step 1: Classify prompt as simple or complex
        is_simple = await simplicity_classifier(user_message)
        if is_simple:
            response = {
                "graph_data": [],
                "summary_answer": f"Simple response: {user_message}",
                "detailed_path": "No decomposition needed for simple prompt.",
                "quality_score": "Satisfactory",
                "processing_time": int((time.time() - start_time) * 1000)
            }
            return jsonify(response)

        # Step 2: Process complex message through AgentGraph
        graph_data, summary_answer, detailed_path, quality_score = await agent_graph.build_task_graph(user_message)

        # Step 3: Determine if further processing is needed
        if not await stop_classifier([summary_answer, detailed_path, quality_score], user_message, detailed_path):
            summary_answer += " (Further processing applied)"
            quality_score = "Good"

        # End processing time
        processing_time = int((time.time() - start_time) * 1000)

        response = {
            "graph_data": graph_data,
            "summary_answer": summary_answer,
            "detailed_path": detailed_path,
            "quality_score": quality_score,
            "processing_time": processing_time
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Internal error in message handling: {e}")
        return handle_internal_error(e)


def create_app():
    """Create Flask app."""
    return app
