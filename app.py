from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

# Import and register the routes
from Routes import setup_routes
setup_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
