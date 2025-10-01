"""Database models for the cycling tools application."""
from __future__ import annotations

from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship


db = SQLAlchemy()


class User(UserMixin, db.Model):
    """Registered athlete who can store and review trainings."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)

    ftp = Column(Float, nullable=True)
    max_hr = Column(Integer, nullable=True)
    weight = Column(Float, nullable=True)

    trainings = relationship("Training", back_populates="user", cascade="all, delete-orphan")
    bikes = relationship("Bike", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<User {self.username}>"


class Bike(db.Model):
    """A bicycle owned by the user for associating with trainings."""

    __tablename__ = "bikes"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(120), nullable=False)
    weight = Column(Float, nullable=False)

    user = relationship("User", back_populates="bikes")
    trainings = relationship("Training", back_populates="bike")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Bike {self.name} ({self.weight} kg)>"


class Training(db.Model):
    """Single analysed training ride stored for a user."""

    __tablename__ = "trainings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    bike_id = Column(Integer, ForeignKey("bikes.id"), nullable=True)

    date = Column(DateTime, nullable=False, default=datetime.utcnow)
    title = Column(String(160), nullable=False)

    distance = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    elevation = Column(Float, nullable=True)
    elevation_loss = Column(Float, nullable=True)

    avg_speed = Column(Float, nullable=True)
    max_speed = Column(Float, nullable=True)
    avg_hr = Column(Float, nullable=True)
    max_hr = Column(Float, nullable=True)
    avg_power = Column(Float, nullable=True)
    max_power = Column(Float, nullable=True)

    normalized_power = Column(Float, nullable=True)
    intensity_factor = Column(Float, nullable=True)
    training_stress_score = Column(Float, nullable=True)
    variability_index = Column(Float, nullable=True)
    calories = Column(Float, nullable=True)
    ftp_used = Column(Float, nullable=True)

    file_path = Column(String(255), nullable=False)

    user = relationship("User", back_populates="trainings")
    bike = relationship("Bike", back_populates="trainings")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Training {self.title} ({self.date:%Y-%m-%d})>"
