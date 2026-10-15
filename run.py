"""
run.py — application entry point.

    python run.py

Creates the Flask app (starts the roadmap sync, opportunity sync and
pseudo-label bootstrap schedulers) and serves it on Config.HOST:Config.PORT.
"""
from careercompass.app import create_app
from careercompass.config import Config

app = create_app()

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
