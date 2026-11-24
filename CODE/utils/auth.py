"""Multi-user authentication, roles and user management for the Flask app.

Usage in app.py:
    from utils.auth import init_auth, login_required, role_required
    init_auth(app)
"""
import hashlib
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import time
from email.message import EmailMessage
from functools import wraps

from flask import (abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = os.environ.get("USERS_DB", "data/users.db")
ROLES = {"admin": "Admin", "hr": "HR analyst", "viewer": "Viewer"}
MIN_PASSWORD = 8
MAX_FAILS, LOCK_SECONDS = 5, 300
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
RESET_MINUTES = 30
_failed = {}  # (username, ip) -> [fail_count, first_fail_time]


# ---------- database ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def _create_tables():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            full_name     TEXT NOT NULL DEFAULT '',
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'viewer',
            active        INTEGER NOT NULL DEFAULT 1,
            created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login    TEXT,
            email         TEXT NOT NULL DEFAULT '',
            reset_requested_at TEXT)""")
        cols = {r[1] for r in con.execute("PRAGMA table_info(users)")}
        if "email" not in cols:
            con.execute("ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT ''")
        if "reset_requested_at" not in cols:
            con.execute("ALTER TABLE users ADD COLUMN reset_requested_at TEXT")
        con.execute("""CREATE TABLE IF NOT EXISTS password_resets (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            token_hash TEXT NOT NULL,
            expires_at REAL NOT NULL,
            used       INTEGER NOT NULL DEFAULT 0)""")
        if con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            env_pw = os.environ.get("ADMIN_PASSWORD")
            pw = env_pw or secrets.token_urlsafe(9)
            con.execute(
                "INSERT INTO users (username, full_name, password_hash, role) "
                "VALUES (?,?,?,?)",
                ("admin", "Administrator", generate_password_hash(pw), "admin"))
            if env_pw:
                print("[auth] First admin created: username 'admin' "
                      "(password from ADMIN_PASSWORD).")
            else:
                print(f"[auth] First admin created: username=admin  password={pw}")
                print("[auth] Log in and change this password on the Account page.")


# ---------- session helpers ----------
def current_user_row():
    uid = session.get("uid")
    if not uid:
        return None
    u = db().execute("SELECT * FROM users WHERE id=? AND active=1",
                     (uid,)).fetchone()
    return u


def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


def check_csrf():
    sent = request.form.get("_csrf", "")
    if not secrets.compare_digest(sent, session.get("_csrf", "")):
        abort(400)


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        u = current_user_row()
        if u is None:
            session.clear()
            if request.path.startswith("/api/"):
                return jsonify(error="login required"), 401
            return redirect(url_for("login", next=request.path))
        g.user = u
        return f(*args, **kwargs)
    return wrapper


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        @login_required
        def wrapper(*args, **kwargs):
            if g.user["role"] not in roles:
                if request.path.startswith("/api/"):
                    return jsonify(error="not allowed for your role"), 403
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ---------- login throttling ----------
def _locked(key):
    rec = _failed.get(key)
    if not rec:
        return False
    if time.time() - rec[1] > LOCK_SECONDS:
        _failed.pop(key, None)
        return False
    return rec[0] >= MAX_FAILS


def _record_fail(key):
    rec = _failed.setdefault(key, [0, time.time()])
    rec[0] += 1


def _safe_next(url):
    return bool(url) and url.startswith("/") and not url.startswith("//")


# ---------- routes ----------
def login():
    if current_user_row():
        return redirect(url_for("overview"))
    error = None
    if request.method == "POST":
        check_csrf()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        key = (username, request.remote_addr)
        if _locked(key):
            error = "Too many failed attempts. Try again in a few minutes."
        else:
            u = db().execute("SELECT * FROM users WHERE username=?",
                             (username,)).fetchone()
            if u and u["active"] and check_password_hash(u["password_hash"], password):
                _failed.pop(key, None)
                nxt = request.args.get("next", "")
                session.clear()
                session["uid"] = u["id"]
                session["user"] = u["username"]
                db().execute("UPDATE users SET last_login=CURRENT_TIMESTAMP "
                             "WHERE id=?", (u["id"],))
                db().commit()
                return redirect(nxt if _safe_next(nxt) else url_for("overview"))
            _record_fail(key)
            error = "Username or password is incorrect."
    return render_template("login.html", error=error)


def logout():
    session.clear()
    return redirect(url_for("login"))


@login_required
def account():
    if request.method == "POST":
        check_csrf()
        cur = request.form.get("current", "")
        new = request.form.get("new", "")
        confirm = request.form.get("confirm", "")
        if not check_password_hash(g.user["password_hash"], cur):
            flash("Current password is incorrect.", "error")
        elif len(new) < MIN_PASSWORD:
            flash(f"New password must be at least {MIN_PASSWORD} characters.", "error")
        elif new != confirm:
            flash("New password and confirmation do not match.", "error")
        else:
            db().execute("UPDATE users SET password_hash=? WHERE id=?",
                         (generate_password_hash(new), g.user["id"]))
            db().commit()
            flash("Password changed.", "ok")
        return redirect(url_for("account"))
    return render_template("account.html", me=g.user)


@role_required("admin")
def users_page():
    rows = db().execute("SELECT * FROM users ORDER BY username").fetchall()
    return render_template("users.html", users=rows, me=g.user)


@role_required("admin")
def users_add():
    check_csrf()
    username = request.form.get("username", "").strip().lower()
    full_name = request.form.get("full_name", "").strip()[:80]
    role = request.form.get("role", "viewer")
    password = request.form.get("password", "")
    email = request.form.get("email", "").strip().lower()
    if not USERNAME_RE.match(username):
        flash("Username must be 3-30 characters: letters, numbers, . _ -", "error")
    elif role not in ROLES:
        flash("Choose a valid role.", "error")
    elif len(password) < MIN_PASSWORD:
        flash(f"Password must be at least {MIN_PASSWORD} characters.", "error")
    elif email and not EMAIL_RE.match(email):
        flash("Enter a valid email address or leave it empty.", "error")
    elif db().execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
        flash(f"The username '{username}' is already taken.", "error")
    elif email and _email_taken(email):
        flash("That email is already used by another account.", "error")
    else:
        db().execute("INSERT INTO users (username, full_name, password_hash, role, email) "
                     "VALUES (?,?,?,?,?)",
                     (username, full_name, generate_password_hash(password), role, email))
        db().commit()
        flash(f"User '{username}' created.", "ok")
    return redirect(url_for("users_page"))


@role_required("admin")
def users_update(uid):
    check_csrf()
    role = request.form.get("role")
    active = 1 if request.form.get("active") == "1" else 0
    if role not in ROLES:
        abort(400)
    target = db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if target is None:
        abort(404)
    if target["role"] == "admin" and (role != "admin" or not active):
        others = db().execute(
            "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1 AND id!=?",
            (uid,)).fetchone()[0]
        if others == 0:
            flash("Keep at least one active admin.", "error")
            return redirect(url_for("users_page"))
    db().execute("UPDATE users SET role=?, active=? WHERE id=?", (role, active, uid))
    db().commit()
    flash(f"Updated '{target['username']}'.", "ok")
    return redirect(url_for("users_page"))


@role_required("admin")
def users_reset(uid):
    check_csrf()
    pw = request.form.get("password", "")
    target = db().execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    if target is None:
        abort(404)
    if len(pw) < MIN_PASSWORD:
        flash(f"Password must be at least {MIN_PASSWORD} characters.", "error")
    else:
        db().execute("UPDATE users SET password_hash=? WHERE id=?",
                     (generate_password_hash(pw), uid))
        db().commit()
        flash(f"Password reset for '{target['username']}'.", "ok")
    return redirect(url_for("users_page"))


# ---------- forgot / reset password ----------
def _email_taken(email, except_id=None):
    row = db().execute("SELECT id FROM users WHERE email!='' AND lower(email)=? "
                       "AND id IS NOT ?", (email.lower(), except_id)).fetchone()
    return row is not None


def _hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _make_reset_link(user_id):
    """Create a one-time link (valid RESET_MINUTES). Only a hash is stored."""
    token = secrets.token_urlsafe(32)
    db().execute("UPDATE password_resets SET used=1 WHERE user_id=? AND used=0",
                 (user_id,))
    db().execute("INSERT INTO password_resets (user_id, token_hash, expires_at) "
                 "VALUES (?,?,?)",
                 (user_id, _hash(token), time.time() + RESET_MINUTES * 60))
    db().commit()
    base = os.environ.get("APP_BASE_URL") or request.host_url
    return f"{base.rstrip('/')}/reset/{token}"


def _smtp_ready():
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def _send_reset_email(to, name, link):
    msg = EmailMessage()
    msg["Subject"] = "Reset your HR Analytics password"
    msg["From"] = os.environ.get("SMTP_FROM") or os.environ["SMTP_USER"]
    msg["To"] = to
    msg.set_content(
        f"Hello {name},\n\n"
        f"Use this link to choose a new password. It works once and expires in "
        f"{RESET_MINUTES} minutes:\n\n{link}\n\n"
        f"If you did not ask for this, you can ignore this email.")
    with smtplib.SMTP(os.environ["SMTP_HOST"],
                      int(os.environ.get("SMTP_PORT", "587")), timeout=15) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        s.send_message(msg)


def forgot():
    if current_user_row():
        return redirect(url_for("account"))
    sent = False
    if request.method == "POST":
        check_csrf()
        sent = True  # same answer for every case, so usernames cannot be guessed
        ident = request.form.get("identity", "").strip().lower()
        key = ("forgot", request.remote_addr, "limit")
        if ident and not _locked(key):
            _record_fail(key)
            u = db().execute(
                "SELECT * FROM users WHERE active=1 AND "
                "(username=? OR (email!='' AND lower(email)=?))",
                (ident, ident)).fetchone()
            if u:
                db().execute("UPDATE users SET reset_requested_at=CURRENT_TIMESTAMP "
                             "WHERE id=?", (u["id"],))
                db().commit()
                if u["email"] and _smtp_ready():
                    try:
                        _send_reset_email(u["email"], u["full_name"] or u["username"],
                                          _make_reset_link(u["id"]))
                    except Exception as exc:  # admin can still create a link
                        print(f"[auth] Could not send reset email: {exc}")
    return render_template("forgot.html", sent=sent,
                           email_enabled=_smtp_ready())


def _valid_reset(token):
    return db().execute(
        "SELECT r.*, u.username FROM password_resets r JOIN users u ON u.id=r.user_id "
        "WHERE r.token_hash=? AND r.used=0 AND r.expires_at>? AND u.active=1",
        (_hash(token), time.time())).fetchone()


def reset(token):
    row = _valid_reset(token)
    if row is None:
        return render_template("reset.html", invalid=True), 400
    if request.method == "POST":
        check_csrf()
        new = request.form.get("new", "")
        confirm = request.form.get("confirm", "")
        if len(new) < MIN_PASSWORD:
            flash(f"Password must be at least {MIN_PASSWORD} characters.", "error")
        elif new != confirm:
            flash("The two passwords do not match.", "error")
        else:
            db().execute("UPDATE users SET password_hash=?, reset_requested_at=NULL "
                         "WHERE id=?", (generate_password_hash(new), row["user_id"]))
            db().execute("UPDATE password_resets SET used=1 WHERE user_id=?",
                         (row["user_id"],))
            db().commit()
            session.clear()
            flash("Password updated. Please sign in.", "ok")
            return redirect(url_for("login"))
        return redirect(url_for("reset", token=token))
    return render_template("reset.html", invalid=False, username=row["username"])


@role_required("admin")
def users_link(uid):
    check_csrf()
    target = db().execute("SELECT * FROM users WHERE id=? AND active=1",
                          (uid,)).fetchone()
    if target is None:
        abort(404)
    link = _make_reset_link(uid)
    db().execute("UPDATE users SET reset_requested_at=NULL WHERE id=?", (uid,))
    db().commit()
    flash(f"Reset link for '{target['username']}' (works once, valid "
          f"{RESET_MINUTES} minutes): {link}", "ok")
    return redirect(url_for("users_page"))


@login_required
def account_email():
    check_csrf()
    email = request.form.get("email", "").strip().lower()
    if email and not EMAIL_RE.match(email):
        flash("Enter a valid email address.", "error")
    elif email and _email_taken(email, g.user["id"]):
        flash("That email is already used by another account.", "error")
    else:
        db().execute("UPDATE users SET email=? WHERE id=?", (email, g.user["id"]))
        db().commit()
        flash("Email saved." if email else "Email removed.", "ok")
    return redirect(url_for("account"))


# ---------- setup ----------
def init_auth(app):
    """Call once after creating the Flask app."""
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
    app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
    _create_tables()

    @app.teardown_appcontext
    def _close_db(exc):
        con = g.pop("db", None)
        if con is not None:
            con.close()

    @app.context_processor
    def _auth_context():
        u = current_user_row() if session.get("uid") else None
        return dict(csrf_token=csrf_token, ROLES=ROLES,
                    current_user=u["username"] if u else None,
                    current_name=(u["full_name"] or u["username"]) if u else None,
                    current_role=u["role"] if u else None)

    rule = app.add_url_rule
    rule("/login", "login", login, methods=["GET", "POST"])
    rule("/logout", "logout", logout)
    rule("/account", "account", account, methods=["GET", "POST"])
    rule("/account/email", "account_email", account_email, methods=["POST"])
    rule("/forgot", "forgot", forgot, methods=["GET", "POST"])
    rule("/reset/<token>", "reset", reset, methods=["GET", "POST"])
    rule("/admin/users/<int:uid>/link", "users_link", users_link, methods=["POST"])
    rule("/admin/users", "users_page", users_page)
    rule("/admin/users/add", "users_add", users_add, methods=["POST"])
    rule("/admin/users/<int:uid>/update", "users_update", users_update, methods=["POST"])
    rule("/admin/users/<int:uid>/reset", "users_reset", users_reset, methods=["POST"])