# app.py is the main entry point for the Flask application. It sets up the Flask app, configures the database, 
# and initializes the SQLAlchemy extension.
# app.py is managed by customer_model.ipynb and if you want to make changes to the app setup, 
# please edit customer_model.ipynb instead of app.py directly.
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app(db_uri: str = "sqlite:///customer_order_insights.db",
               secret_key: str = "9cbd5ad3-cc37-4515-924f-45afc2932d46"):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = secret_key
    app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # Flask 3.x compatible: do one-time setup here
    with app.app_context():
        db.create_all()

    return app
