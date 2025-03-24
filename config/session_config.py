import os
from flask import Flask
from flask_session import Session

def configure_session(app: Flask):
    app.secret_key = 'demo_secret_key'
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flask_session")
    app.config["SESSION_PERMANENT"] = False
    Session(app) 