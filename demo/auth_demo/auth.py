#!/usr/bin/env python3
"""A small user authentication system backed by SQLite.

Only the Python standard library is used. Passwords are stored as salted
PBKDF2-HMAC-SHA256 hashes; every user gets an independent random salt.

Usage:
    python3 auth.py register <username>
    python3 auth.py login <username>

The password is always read interactively with getpass, never from argv.
"""

import getpass
import hashlib
import hmac
import os
import secrets
import sqlite3
import sys

# --- Hashing parameters -----------------------------------------------------

PBKDF2_ITERATIONS = 200_000
HASH_ALGORITHM = "sha256"
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 8

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


# --- Exceptions -------------------------------------------------------------


class AuthError(Exception):
    """Base class for authentication errors."""


class UserAlreadyExistsError(AuthError):
    """Raised when registering a username that is already taken."""


class ValidationError(AuthError, ValueError):
    """Raised when a username or password fails validation."""


# --- Database helpers -------------------------------------------------------


def get_db_path(db_path=None):
    """Resolve the database path.

    Priority: explicit argument > AUTH_DB_PATH env var > users.db next to this file.
    """
    if db_path:
        return db_path
    return os.environ.get("AUTH_DB_PATH") or DEFAULT_DB_PATH


def _connect(db_path=None):
    conn = sqlite3.connect(get_db_path(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            iterations INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    return conn


# --- Hashing helpers --------------------------------------------------------


def _hash_password(password, salt, iterations=PBKDF2_ITERATIONS, algorithm=HASH_ALGORITHM):
    return hashlib.pbkdf2_hmac(algorithm, password.encode("utf-8"), salt, iterations)


def _validate_credentials(username, password):
    if not isinstance(username, str) or not username.strip():
        raise ValidationError("Username must not be empty.")
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError(
            "Password must be at least %d characters long." % MIN_PASSWORD_LENGTH
        )


# --- Public API -------------------------------------------------------------


def register(username, password, db_path=None):
    """Register a new user.

    Returns the new user's id. Raises ValidationError for bad input and
    UserAlreadyExistsError if the username is taken.
    """
    _validate_credentials(username, password)
    username = username.strip()

    salt = secrets.token_bytes(SALT_BYTES)
    password_hash = _hash_password(password, salt)

    conn = _connect(db_path)
    try:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, salt, password_hash, iterations, algorithm)"
                " VALUES (?, ?, ?, ?, ?)",
                (username, salt, password_hash, PBKDF2_ITERATIONS, HASH_ALGORITHM),
            )
        except sqlite3.IntegrityError:
            raise UserAlreadyExistsError("Username %r is already taken." % username)
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def login(username, password, db_path=None):
    """Return True if the credentials are valid, False otherwise.

    A missing user and a wrong password are indistinguishable to the caller.
    """
    if not isinstance(username, str) or not isinstance(password, str):
        return False

    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT salt, password_hash, iterations, algorithm FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        # Perform an equivalent hash computation so that a missing user does
        # not stand out through timing.
        _hash_password(password, b"\x00" * SALT_BYTES)
        return False

    salt, stored_hash, iterations, algorithm = row
    candidate = _hash_password(password, salt, iterations, algorithm)
    return hmac.compare_digest(candidate, stored_hash)


# --- Command line interface -------------------------------------------------

USAGE = """Usage:
  python3 auth.py register <username>
  python3 auth.py login <username>

The password is read interactively (never passed on the command line)."""


def _cmd_register(username):
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Error: passwords do not match.", file=sys.stderr)
        return 1
    try:
        user_id = register(username, password)
    except UserAlreadyExistsError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    except ValidationError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    except sqlite3.Error as exc:
        print("Database error: %s" % exc, file=sys.stderr)
        return 1
    print("User %r registered successfully (id=%d)." % (username.strip(), user_id))
    return 0


def _cmd_login(username):
    password = getpass.getpass("Password: ")
    try:
        ok = login(username, password)
    except sqlite3.Error as exc:
        print("Database error: %s" % exc, file=sys.stderr)
        return 1
    if ok:
        print("Login successful. Welcome, %s!" % username.strip())
        return 0
    print("Login failed: invalid username or password.", file=sys.stderr)
    return 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("register", "login"):
        print(USAGE, file=sys.stderr)
        return 2

    command, username = argv
    if command == "register":
        return _cmd_register(username)
    return _cmd_login(username)


if __name__ == "__main__":
    sys.exit(main())
