"""WTForms definitions for authentication."""
from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import (
    EmailField,
    FloatField,
    IntegerField,
    PasswordField,
    StringField,
    SubmitField,
)
from wtforms.validators import DataRequired, Email, Length, NumberRange


class LoginForm(FlaskForm):
    """Simple login form requesting email and password."""

    email = EmailField("E-mail", validators=[DataRequired(), Email()])
    password = PasswordField("Heslo", validators=[DataRequired()])
    submit = SubmitField("Přihlásit se")


class RegisterForm(FlaskForm):
    """Registration form collecting username, email and password."""

    username = StringField("Uživatelské jméno", validators=[DataRequired(), Length(min=3, max=80)])
    email = EmailField("E-mail", validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField("Heslo", validators=[DataRequired(), Length(min=6)])
    submit = SubmitField("Registrovat se")


class ProfileForm(FlaskForm):
    """Collects athlete-specific physiology inputs."""

    ftp = FloatField("FTP (W)", validators=[DataRequired(), NumberRange(min=50, message="FTP musí být kladné.")])
    max_hr = IntegerField(
        "Maximální tep (bpm)", validators=[DataRequired(), NumberRange(min=60, max=240, message="Zadejte reálnou hodnotu.")]
    )
    weight = FloatField(
        "Váha (kg)",
        validators=[DataRequired(), NumberRange(min=30, max=200, message="Váha musí být v rozumném rozmezí.")],
    )
    submit = SubmitField("Uložit profil")


class BikeForm(FlaskForm):
    """Form for adding a bicycle to the athlete profile."""

    name = StringField("Název kola", validators=[DataRequired(), Length(max=120)])
    weight = FloatField(
        "Hmotnost kola (kg)", validators=[DataRequired(), NumberRange(min=2, max=25, message="Zadejte hmotnost v kilogramech.")]
    )
    submit = SubmitField("Přidat kolo")
