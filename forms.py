"""WTForms definitions for authentication."""
from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import EmailField, PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, Length


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
