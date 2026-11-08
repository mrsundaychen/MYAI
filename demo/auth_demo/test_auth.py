"""Tests for auth.py. Run with: python3 -m unittest discover -v"""

import os
import sqlite3
import tempfile
import unittest

import auth


class AuthTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.db_path)  # let sqlite create it fresh

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _fetch(self, username):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                "SELECT salt, password_hash, iterations, algorithm FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        finally:
            conn.close()

    # --- required cases -----------------------------------------------------

    def test_register_success(self):
        user_id = auth.register("alice", "correct horse", db_path=self.db_path)
        self.assertIsInstance(user_id, int)
        self.assertIsNotNone(self._fetch("alice"))

    def test_register_duplicate_raises(self):
        auth.register("alice", "correct horse", db_path=self.db_path)
        with self.assertRaises(auth.UserAlreadyExistsError):
            auth.register("alice", "another password", db_path=self.db_path)

    def test_login_success(self):
        auth.register("alice", "correct horse", db_path=self.db_path)
        self.assertTrue(auth.login("alice", "correct horse", db_path=self.db_path))

    def test_login_wrong_password(self):
        auth.register("alice", "correct horse", db_path=self.db_path)
        self.assertFalse(auth.login("alice", "wrong horse", db_path=self.db_path))

    def test_short_password_raises(self):
        with self.assertRaises(ValueError):
            auth.register("alice", "short", db_path=self.db_path)

    # --- extra cases --------------------------------------------------------

    def test_login_unknown_user(self):
        self.assertFalse(auth.login("nobody", "correct horse", db_path=self.db_path))

    def test_empty_username_raises(self):
        with self.assertRaises(ValueError):
            auth.register("", "correct horse", db_path=self.db_path)
        with self.assertRaises(ValueError):
            auth.register("   ", "correct horse", db_path=self.db_path)

    def test_username_is_stripped(self):
        auth.register("  alice  ", "correct horse", db_path=self.db_path)
        self.assertTrue(auth.login("alice", "correct horse", db_path=self.db_path))

    def test_password_not_stored_in_plaintext(self):
        password = "correct horse"
        auth.register("alice", password, db_path=self.db_path)
        salt, password_hash, iterations, algorithm = self._fetch("alice")
        self.assertNotEqual(password_hash, password.encode("utf-8"))
        self.assertNotIn(password.encode("utf-8"), password_hash)
        self.assertGreaterEqual(iterations, 100_000)
        self.assertEqual(algorithm, "sha256")
        self.assertEqual(len(salt), auth.SALT_BYTES)

    def test_salts_are_random(self):
        auth.register("alice", "same password", db_path=self.db_path)
        auth.register("bob", "same password", db_path=self.db_path)
        alice = self._fetch("alice")
        bob = self._fetch("bob")
        self.assertNotEqual(alice[0], bob[0])
        self.assertNotEqual(alice[1], bob[1])

    def test_db_path_from_env(self):
        old = os.environ.get("AUTH_DB_PATH")
        os.environ["AUTH_DB_PATH"] = self.db_path
        try:
            auth.register("alice", "correct horse")
            self.assertTrue(auth.login("alice", "correct horse"))
        finally:
            if old is None:
                os.environ.pop("AUTH_DB_PATH", None)
            else:
                os.environ["AUTH_DB_PATH"] = old


if __name__ == "__main__":
    unittest.main()
