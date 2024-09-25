from api.server_api import create_app
from api.client import app as dash_app

# Create Flask application
flask_app = create_app()

# Integrate Dash app with Flask app
dash_app.init_app(flask_app)

# Run Flask application (which now includes Dash)
if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5000, debug=True)