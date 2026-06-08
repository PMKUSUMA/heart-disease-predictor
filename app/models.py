from flask_sqlalchemy import SQLAlchemy

db: SQLAlchemy = SQLAlchemy()


class PredictionLog(db.Model):
    id: int = db.Column(db.Integer, primary_key=True)
    age: float = db.Column(db.Float, nullable=False)
    sex: float = db.Column(db.Float, nullable=False)
    prediction: int = db.Column(db.Integer, nullable=False)
    confidence: float = db.Column(db.Float, nullable=False)
    timestamp: str = db.Column(db.DateTime, server_default=db.func.now())