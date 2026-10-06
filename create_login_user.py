import os
import re
import sqlite3

from werkzeug.security import generate_password_hash


# -------------------------------------------------
# PROJECT DATABASE
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_FILE = os.path.join(DATABASE_DIR, "users.db")

os.makedirs(DATABASE_DIR, exist_ok=True)


# -------------------------------------------------
# EMAIL VALIDATION
# -------------------------------------------------

EMAIL_RE = re.compile(
    r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
)


# -------------------------------------------------
# GET EMAIL
# -------------------------------------------------

email = input("Login email: ").strip().lower()

if not EMAIL_RE.fullmatch(email):
    print("\nInvalid email address.")
    input("Press Enter to exit...")
    raise SystemExit


# -------------------------------------------------
# GET PASSWORD
# -------------------------------------------------

print()
print("Enter your password below.")
print("The password will be visible while typing.")
print()

password = input("Login password: ")


# -------------------------------------------------
# CHECK PASSWORD
# -------------------------------------------------

if not password:
    print("\nPassword cannot be empty.")
    input("Press Enter to exit...")
    raise SystemExit


confirm = input("Confirm password: ")


if password != confirm:
    print("\nPasswords do not match.")
    input("Press Enter to exit...")
    raise SystemExit


# -------------------------------------------------
# GET NAME
# -------------------------------------------------

name = input("Name: ").strip()

if not name:
    name = email.split("@")[0]


# -------------------------------------------------
# CONNECT DATABASE
# -------------------------------------------------

connection = sqlite3.connect(DATABASE_FILE)

try:

    # Create users table if it doesn't exist
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Hash password before saving
    hashed_password = generate_password_hash(password)


    # Check whether email already exists
    existing_user = connection.execute(
        """
        SELECT id
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()


    if existing_user:

        # Update existing account
        connection.execute(
            """
            UPDATE users
            SET name = ?,
                password = ?
            WHERE email = ?
            """,
            (
                name,
                hashed_password,
                email
            )
        )

        print()
        print("====================================")
        print("Login account updated successfully.")
        print("====================================")

    else:

        # Create new account
        connection.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                password
            )
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                hashed_password
            )
        )

        print()
        print("====================================")
        print("Login account created successfully.")
        print("====================================")


    connection.commit()


finally:

    connection.close()


print()
print("Email:", email)
print("You can now use this email and password to login.")
print()

input("Press Enter to exit...")