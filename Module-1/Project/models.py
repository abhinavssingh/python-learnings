# models.py defines the database models for the application using SQLAlchemy. 
# It includes the Customer model which represents a customer order.
# please edit customer_model.ipynb instead of models.py directly.
from datetime import datetime

from app import db  # import the shared SQLAlchemy() instance


class Customer(db.Model):
    __tablename__ = "customers"

    id         = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(120), nullable=False)
    last_name  = db.Column(db.String(120), nullable=False)
    email      = db.Column(db.String(255), index=True)
    phone      = db.Column(db.String(30))
    product    = db.Column(db.String(255), nullable=False)
    category   = db.Column(db.String(100), nullable=False)
    price      = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Customer id={self.id} name={self.first_name} {self.last_name} email={self.email} product={self.product} category={self.category} price={self.price}>"
