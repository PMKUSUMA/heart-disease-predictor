import os

from flask import Flask
from app.models import db


def create_app() -> Flask:
    app: Flask = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "heart-disease-predictor-secret")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictions.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    with app.app_context():
        db.create_all()

    from app.routes import main
    app.register_blueprint(main)

    return app