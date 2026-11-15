# ============================================================
# COLLEGE PREDICTOR SYSTEM
# Flask + Machine Learning
# KCET + EAMCET
# ============================================================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify
)

import pandas as pd
import os
import json
import urllib.request
import urllib.error
import joblib
import psycopg
from psycopg.rows import dict_row
import logging
import traceback
import re
import secrets
from datetime import datetime, timedelta

from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from flask_mail import Mail, Message

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from functools import wraps
from werkzeug.utils import secure_filename

from flask_wtf.csrf import CSRFProtect

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from authlib.integrations.flask_client import OAuth


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ============================================================
# EMAIL CONFIGURATION
# ============================================================

app.config["MAIL_SERVER"] = "smtp.gmail.com"

app.config["MAIL_PORT"] = 587

app.config["MAIL_USE_TLS"] = True

app.config["MAIL_USERNAME"] = os.getenv(
    "MAIL_USERNAME"
)

app.config["MAIL_PASSWORD"] = os.getenv(
    "MAIL_PASSWORD"
)

app.config["MAIL_DEFAULT_SENDER"] = os.getenv(
    "MAIL_USERNAME"
)

mail = Mail(app)



# ============================================================
# SECURITY CONFIGURATION
# ============================================================

SECRET_KEY = os.getenv(
    "SECRET_KEY"
)

if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY is missing from .env file."
    )

if len(SECRET_KEY) < 32:
    raise RuntimeError(
        "SECRET_KEY must contain at least 32 characters."
    )


ADMIN_EMAIL = os.getenv(
    "ADMIN_EMAIL",
    "udayu6853@gmail.com"
).strip().lower()

app.config["SECRET_KEY"] = SECRET_KEY

app.config["SESSION_COOKIE_HTTPONLY"] = True

app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.config["SESSION_COOKIE_SECURE"] = (
    os.getenv(
        "SESSION_COOKIE_SECURE",
        "False"
    ).lower()
    == "true"
)

app.config["SESSION_COOKIE_PATH"] = "/"

app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)

app.config["MAX_CONTENT_LENGTH"] = (
    5 * 1024 * 1024
)

# ============================================================
# PROFILE IMAGE UPLOAD CONFIGURATION
# ============================================================

PROFILE_UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "static",
    "uploads",
    "profiles"
)

os.makedirs(
    PROFILE_UPLOAD_FOLDER,
    exist_ok=True
)

ALLOWED_PROFILE_IMAGE_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

# ============================================================
# GOOGLE OAUTH CONFIGURATION
# ============================================================

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI",
    "http://127.0.0.1:5000/google/callback"
)

if not GOOGLE_CLIENT_ID:
    raise RuntimeError(
        "GOOGLE_CLIENT_ID is missing from .env file."
    )

if not GOOGLE_CLIENT_SECRET:
    raise RuntimeError(
        "GOOGLE_CLIENT_SECRET is missing from .env file."
    )

oauth = OAuth(app)

google = oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url=(
        "https://accounts.google.com/"
        ".well-known/openid-configuration"
    ),
    client_kwargs={
        "scope": "openid email profile"
    }
)


# ============================================================
# CSRF PROTECTION
# ============================================================

csrf = CSRFProtect(app)


# ============================================================
# RATE LIMITING
# ============================================================

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[]
)


# ============================================================
# LOGGING
# ============================================================

LOG_DIRECTORY = "logs"

os.makedirs(
    LOG_DIRECTORY,
    exist_ok=True
)

log_file = os.path.join(
    LOG_DIRECTORY,
    "app.log"
)

file_handler = RotatingFileHandler(
    log_file,
    maxBytes=5 * 1024 * 1024,
    backupCount=5
)

file_handler.setLevel(
    logging.INFO
)

formatter = logging.Formatter(
    "%(asctime)s | "
    "%(levelname)s | "
    "%(message)s"
)

file_handler.setFormatter(
    formatter
)

app.logger.addHandler(
    file_handler
)

app.logger.setLevel(
    logging.INFO
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

KCET_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "kcet_colleges.csv"
)

EAMCET_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "eamcet_colleges.csv"
)

# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = os.path.join(
    BASE_DIR,
    "model",
    "college_predictor.pkl"
)

PERFORMANCE_FILE = os.path.join(
    BASE_DIR,
    "model",
    "model_performance.pkl"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_FILE),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(PERFORMANCE_FILE),
    exist_ok=True
)

# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_numeric(
    value,
    default=0
):

    try:

        if pd.isna(value):
            return default

        value = str(
            value
        ).strip()

        if not value:
            return default

        value = value.replace(
            ",",
            ""
        )

        value = re.sub(
            r"[₹$]",
            "",
            value
        )

        value = re.sub(
            r"[^\d.\-]",
            "",
            value
        )

        if not value:
            return default

        return float(
            value
        )

    except (
        ValueError,
        TypeError
    ):

        return default


def clean_integer(
    value,
    default=0
):

    try:

        number = clean_numeric(
            value,
            default
        )

        return int(
            float(number)
        )

    except (
        ValueError,
        TypeError
    ):

        return default


def clean_text(
    value,
    default=""
):

    try:

        if pd.isna(value):
            return default

        value = str(
            value
        ).strip()

        return value or default

    except Exception:

        return default


def clean_image(
    value
):

    value = clean_text(
        value
    )

    if not value:
        return "/static/images/default.jpg"

    if value.lower() in [
        "nan",
        "none",
        "null",
        "n/a",
        "na"
    ]:
        return "/static/images/default.jpg"

    return value


def clean_rating(
    value
):

    rating = clean_numeric(
        value,
        0
    )

    if rating < 0:
        rating = 0

    if rating > 5:
        rating = 5

    return round(
        rating,
        2
    )


def clean_rank(
    value
):

    rank = clean_integer(
        value,
        0
    )

    if rank < 0:
        rank = 0

    return rank


def clean_branch(
    value
):

    value = clean_text(
        value
    ).upper()

    replacements = {
        "COMPUTER SCIENCE": "CSE",
        "COMPUTER SCIENCE AND ENGINEERING": "CSE",
        "COMPUTER SCIENCE & ENGINEERING": "CSE",
        "ARTIFICIAL INTELLIGENCE": "AIML",
        "ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING": "AIML",
        "INFORMATION SCIENCE": "ISE",
        "INFORMATION SCIENCE AND ENGINEERING": "ISE",
        "ELECTRONICS AND COMMUNICATION": "ECE",
        "ELECTRONICS AND COMMUNICATION ENGINEERING": "ECE"
    }

    return replacements.get(
        value,
        value
    )


def clean_exam(
    value
):

    value = clean_text(
        value
    ).upper()

    if value == "KCET":
        return "KCET"

    if value == "EAMCET":
        return "EAMCET"

    return value


def clean_website(
    value,
    college_name=""
):

    value = clean_text(
        value
    )

    if value.lower() in [
        "nan",
        "none",
        "null",
        "n/a",
        "na"
    ]:
        value = ""

    # Remove markdown-link format if present
    markdown_match = re.match(
        r"\[([^\]]+)\]\(([^)]+)\)",
        value
    )

    if markdown_match:
        value = markdown_match.group(2)

    if value:

        value = value.strip()

        if not value.startswith(
            (
                "http://",
                "https://"
            )
        ):
            value = (
                "https://"
                + value
            )

        return value

    # Known college websites
    known_websites = {

        "RNS Institute of Technology":
            "https://www.rnsit.ac.in/",

        "R.V. College of Engineering":
            "https://www.rvce.edu.in/",

        "BMS College of Engineering":
            "https://www.bmsce.ac.in/",

        "PES University":
            "https://pes.edu/",

        "New Horizon College of Engineering":
            "https://newhorizoncollegeofengineering.in/",

        "Dayananda Sagar College of Engineering":
            "https://www.dsce.edu.in/",

        "Bangalore Institute of Technology":
            "https://bit-bangalore.edu.in/"
    }

    if college_name in known_websites:

        return known_websites[
            college_name
        ]

    return "#"


def clean_rank_for_dataset(
    value
):

    rank = clean_rank(
        value
    )

    return rank


# ============================================================
# VALIDATION
# ============================================================

ALLOWED_EXAMS = {
    "KCET",
    "EAMCET"
}

ALLOWED_BRANCHES = {
    "CSE",
    "AIML",
    "ISE",
    "ECE"
}


def is_valid_email(
    email
):

    pattern = (
        r"^[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}$"
    )

    return bool(
        re.match(
            pattern,
            email
        )
    )


def validate_password(
    password
):

    if not password:
        return (
            False,
            "Password cannot be empty."
        )

    if len(password) < 8:
        return (
            False,
            "Password must contain at least 8 characters."
        )

    if len(password) > 128:
        return (
            False,
            "Password is too long."
        )

    if not re.search(
        r"[A-Za-z]",
        password
    ):
        return (
            False,
            "Password must contain at least one letter."
        )

    if not re.search(
        r"\d",
        password
    ):
        return (
            False,
            "Password must contain at least one number."
        )

    return (
        True,
        ""
    )


def validate_prediction_input(
    exam,
    rank,
    branch
):

    if exam not in ALLOWED_EXAMS:

        return (
            False,
            "Invalid entrance exam."
        )

    if branch not in ALLOWED_BRANCHES:

        return (
            False,
            "Invalid branch."
        )

    try:

        rank = int(
            rank
        )

    except (
        ValueError,
        TypeError
    ):

        return (
            False,
            "Rank must be a valid number."
        )

    if rank <= 0:

        return (
            False,
            "Rank must be greater than zero."
        )

    if rank > 10000000:

        return (
            False,
            "Rank is too large."
        )

    return (
        True,
        ""
    )


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_dataset(
    dataframe,
    exam_name
):

    df = dataframe.copy()

    # Remove unnamed columns
    df = df.loc[
        :,
        ~df.columns.str.contains(
            "^Unnamed",
            case=False
        )
    ]

    # Strip column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Required columns
    if "College Name" not in df.columns:
        df["College Name"] = "Unknown College"

    if "Branch" not in df.columns:
        df["Branch"] = ""

    if "Cutoff Rank" not in df.columns:
        df["Cutoff Rank"] = 0

    # Normalize KCET fee column
    if (
        "Kcet Fees" not in df.columns
        and
        "KCET Fees" in df.columns
    ):

        df["Kcet Fees"] = df[
            "KCET Fees"
        ]

    # Normalize EAMCET fee column
    if (
        "Eamcet Fees" not in df.columns
        and
        "EAMCET Fees" in df.columns
    ):

        df["Eamcet Fees"] = df[
            "EAMCET Fees"
        ]

    # Text columns
    for column in [
        "College Name",
        "Branch",
        "Location",
        "State",
        "Website",
        "Image"
    ]:

        if column in df.columns:

            df[column] = df[
                column
            ].apply(
                clean_text
            )

    # Numeric columns
    for column in [
        "Management Fees",
        "Kcet Fees",
        "KCET Fees",
        "Eamcet Fees",
        "EAMCET Fees",
        "Hostel Fees"
    ]:

        if column in df.columns:

            df[column] = df[
                column
            ].apply(
                clean_numeric
            )

    # Branch
    df["Branch"] = df[
        "Branch"
    ].apply(
        clean_branch
    )

    # Cutoff
    df["Cutoff Rank"] = df[
        "Cutoff Rank"
    ].apply(
        clean_rank_for_dataset
    )

    # Rating
    if "Rating" in df.columns:

        df["Rating"] = df[
            "Rating"
        ].apply(
            clean_rating
        )

    else:

        df["Rating"] = 0.0

    # Website
    if "Website" in df.columns:

        df["Website"] = df.apply(

            lambda row:
            clean_website(
                row.get(
                    "Website",
                    ""
                ),
                row.get(
                    "College Name",
                    ""
                )
            ),

            axis=1
        )

    else:

        df["Website"] = df[
            "College Name"
        ].apply(
            lambda name:
            clean_website(
                "",
                name
            )
        )

    # Image
    if "Image" in df.columns:

        df["Image"] = df[
            "Image"
        ].apply(
            clean_image
        )

    else:

        df["Image"] = (
            "/static/images/default.jpg"
        )

    # Exam
    df["Exam"] = exam_name

    # Remove invalid cutoff
    df = df[
        df["Cutoff Rank"] > 0
    ]

    # Remove duplicate colleges
    df = df.drop_duplicates(
        subset=[
            "College Name",
            "Branch",
            "Cutoff Rank"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# LOAD DATASETS
# ============================================================

try:

    if not os.path.exists(
        KCET_FILE
    ):
        raise FileNotFoundError(
            KCET_FILE
        )

    if not os.path.exists(
        EAMCET_FILE
    ):
        raise FileNotFoundError(
            EAMCET_FILE
        )

    kcet_df = pd.read_csv(
        KCET_FILE
    )

    eamcet_df = pd.read_csv(
        EAMCET_FILE
    )

    kcet_df = prepare_dataset(
        kcet_df,
        "KCET"
    )

    eamcet_df = prepare_dataset(
        eamcet_df,
        "EAMCET"
    )

    app.logger.info(
        "KCET records loaded: %s",
        len(kcet_df)
    )

    app.logger.info(
        "EAMCET records loaded: %s",
        len(eamcet_df)
    )

except Exception as error:

    app.logger.error(
        "Dataset loading failed: %s",
        error
    )

    kcet_df = pd.DataFrame()

    eamcet_df = pd.DataFrame()

# DATABASE CONNECTION
# ============================================================
import sqlite3

class CursorWrapper:
    def __init__(self, cur, is_returning=False):
        self._cur = cur
        if is_returning:
            self._rows = cur.fetchall()
            self._idx = 0
        else:
            self._rows = None

    def fetchone(self):
        if self._rows is not None:
            if self._idx < len(self._rows):
                res = self._rows[self._idx]
                self._idx += 1
                return res
            return None
        return self._cur.fetchone()

    def fetchall(self):
        if self._rows is not None:
            res = self._rows[self._idx:]
            self._idx = len(self._rows)
            return res
        return self._cur.fetchall()

def sqlite_dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row))

class SQLiteConnectionWrapper:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite_dict_factory

    def execute(self, sql, params=None):
        translated_sql = sql.replace("%s", "?")
        translated_sql = re.sub(
            r'INTEGER\s+GENERATED\s+BY\s+DEFAULT\s+AS\s+IDENTITY\s+PRIMARY\s+KEY',
            'INTEGER PRIMARY KEY AUTOINCREMENT',
            translated_sql,
            flags=re.IGNORECASE
        )
        is_alter = "add column if not exists" in translated_sql.lower()
        if is_alter:
            translated_sql = re.sub(
                r'ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS',
                'ADD COLUMN',
                translated_sql,
                flags=re.IGNORECASE
            )
        cur = self.conn.cursor()
        try:
            if params is not None:
                cur.execute(translated_sql, params)
            else:
                cur.execute(translated_sql)
        except sqlite3.OperationalError as e:
            if is_alter and "duplicate column name" in str(e).lower():
                pass
            else:
                raise
        return CursorWrapper(cur, is_returning="returning" in translated_sql.lower())

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()

DATABASE_URL = os.getenv("DATABASE_URL")
USE_LOCAL_DB = os.getenv("USE_LOCAL_DB", "false").strip().lower() in ("true", "1", "yes")
_USE_SQLITE_FALLBACK = USE_LOCAL_DB

def get_db_connection():
    global _USE_SQLITE_FALLBACK
    if DATABASE_URL and not _USE_SQLITE_FALLBACK:
        try:
            connection = psycopg.connect(
                DATABASE_URL,
                row_factory=dict_row,
                connect_timeout=3
            )
            return connection
        except Exception:
            _USE_SQLITE_FALLBACK = True
            app.logger.info("Cloud PostgreSQL unreachable. Using local SQLite database (database/college_predictor.db)")

    db_dir = os.path.join(os.path.dirname(__file__), "database")
    os.makedirs(db_dir, exist_ok=True)
    local_db = os.path.join(db_dir, "college_predictor.db")
    return SQLiteConnectionWrapper(local_db)

# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_database():

    connection = get_db_connection()

    try:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users
            (
                id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                google_id TEXT,
                phone TEXT,
                profile_image TEXT,
                role TEXT DEFAULT 'user'
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS email_verifications
            (
                id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                otp_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_colleges
            (
                id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,

                user_id INTEGER NOT NULL,

                exam TEXT NOT NULL,

                college_index INTEGER NOT NULL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(user_id)
                REFERENCES users(id)
                ON DELETE CASCADE,

                UNIQUE(
                    user_id,
                    exam,
                    college_index
                )
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history
            (
                id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,

                user_id INTEGER NOT NULL,

                exam TEXT NOT NULL,

                rank INTEGER NOT NULL,

                branch TEXT NOT NULL,

                category TEXT DEFAULT '',

                gender TEXT DEFAULT '',

                preferred_location TEXT DEFAULT '',

                state TEXT DEFAULT '',

                fee_range TEXT DEFAULT '',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
            )
            """
        )

        try:
            demo_user = connection.execute(
                "SELECT id FROM users WHERE LOWER(email) = %s",
                ("student@smartpredictor.com",)
            ).fetchone()
            if not demo_user:
                connection.execute(
                    """
                    INSERT INTO users (name, email, password)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (email) DO NOTHING
                    """,
                    (
                        "Student Demo",
                        "student@smartpredictor.com",
                        generate_password_hash("Student@123")
                    )
                )
            uday_user = connection.execute(
                "SELECT id FROM users WHERE LOWER(email) = %s",
                ("udayu6853@gmail.com",)
            ).fetchone()
            if not uday_user:
                connection.execute(
                    """
                    INSERT INTO users (name, email, password)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (email) DO NOTHING
                    """,
                    (
                        "Uday",
                        "udayu6853@gmail.com",
                        generate_password_hash("Alludu@2004")
                    )
                )
        except Exception:
            pass

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


init_database()

# ============================================================
# DATABASE COMPATIBILITY MIGRATION
# ============================================================

def migrate_database():

    connection = get_db_connection()

    try:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS email_verifications
            (
                id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                otp_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            ALTER TABLE prediction_history
            ADD COLUMN IF NOT EXISTS category TEXT DEFAULT ''
            """
        )

        connection.execute(
            """
            ALTER TABLE prediction_history
            ADD COLUMN IF NOT EXISTS gender TEXT DEFAULT ''
            """
        )

        connection.execute(
            """
            ALTER TABLE prediction_history
            ADD COLUMN IF NOT EXISTS preferred_location TEXT DEFAULT ''
            """
        )

        connection.execute(
            """
            ALTER TABLE prediction_history
            ADD COLUMN IF NOT EXISTS state TEXT DEFAULT ''
            """
        )

        connection.execute(
            """
            ALTER TABLE prediction_history
            ADD COLUMN IF NOT EXISTS fee_range TEXT DEFAULT ''
            """
        )

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


migrate_database()

# ============================================================
# GOOGLE AUTH DATABASE MIGRATION
# ============================================================

def migrate_google_auth():

    connection = get_db_connection()

    try:

        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS google_id TEXT
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_google_id
            ON users(google_id)
            """
        )

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


migrate_google_auth()

# ============================================================
# PROFILE FIELD MIGRATION
# ============================================================

def migrate_profile_fields():

    connection = get_db_connection()

    try:
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS phone TEXT
            """
        )

        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS profile_image TEXT
            """
        )

        connection.commit()

        app.logger.info(
            "Profile fields migration completed."
        )

    except Exception:
        connection.rollback()
        app.logger.exception(
            "Profile fields migration failed."
        )
        raise

    finally:
        connection.close()


migrate_profile_fields()


# ============================================================
# ROLE-BASED ACCESS CONTROL (RBAC) MIGRATION
# ============================================================

def migrate_user_roles():

    connection = get_db_connection()

    try:
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'user'
            """
        )
        connection.commit()
    except Exception:
        pass

    try:
        connection.execute(
            """
            UPDATE users
            SET role = 'admin'
            WHERE LOWER(email) = %s
            """,
            (ADMIN_EMAIL,)
        )
        connection.commit()
        app.logger.info("User role RBAC migration completed successfully.")
    except Exception:
        pass
    finally:
        try:
            connection.close()
        except Exception:
            pass


migrate_user_roles()


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = None

model_performance = {}

try:

    if os.path.exists(
        MODEL_FILE
    ):

        model = joblib.load(
            MODEL_FILE
        )

        app.logger.info(
            "Machine learning model loaded successfully."
        )

    else:

        app.logger.warning(
            "ML model not found: %s",
            MODEL_FILE
        )

except Exception as error:

    app.logger.error(
        "ML model loading failed: %s",
        error
    )


try:

    if os.path.exists(
        PERFORMANCE_FILE
    ):

        model_performance = joblib.load(
            PERFORMANCE_FILE
        )

except Exception as error:

    app.logger.warning(
        "Model performance file could not be loaded: %s",
        error
    )


# ============================================================
# LOGIN REQUIRED DECORATOR
# ============================================================

def login_required(
    function
):

    @wraps(function)
    def wrapper(
        *args,
        **kwargs
    ):

        user_id = session.get(
            "user_id"
        )

        if not user_id:

            return redirect(
                url_for(
                    "login"
                )
            )

        try:

            connection = (
                get_db_connection()
            )

            user = connection.execute(

                """
                SELECT id, email, role
                FROM users
                WHERE id = %s
                """,

                (
                    user_id,
                )

            ).fetchone()

            connection.close()

            if user is None:

                session.clear()

                return redirect(
                    url_for(
                        "login"
                    )
                )

            user_role = (user.get("role") or ("admin" if (user.get("email") or "").lower() == ADMIN_EMAIL else "user")).lower().strip()
            user_email = (user.get("email") or "").lower().strip()
            is_admin = (user_role == "admin" or user_email == ADMIN_EMAIL)
            session["user_role"] = "admin" if is_admin else "user"
            session["is_admin"] = is_admin

        except Exception as error:

            app.logger.error(
                "Session validation error: %s",
                error
            )

            session.clear()

            return redirect(
                url_for(
                    "login"
                )
            )

        return function(
            *args,
            **kwargs
        )

    return wrapper


# ============================================================
# SYSTEM ADMINISTRATOR AUTHORIZATION DECORATOR (RBAC)
# ============================================================

def admin_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            flash("Please sign in as System Administrator to access this area.", "warning")
            return redirect(url_for("login"))

        try:
            connection = get_db_connection()
            user = connection.execute(
                """
                SELECT id, email, role
                FROM users
                WHERE id = %s
                """,
                (user_id,)
            ).fetchone()
            connection.close()

            if user is None:
                session.clear()
                flash("Session expired. Please sign in again.", "warning")
                return redirect(url_for("login"))

            user_role = (user.get("role") or ("admin" if (user.get("email") or "").lower() == ADMIN_EMAIL else "user")).lower().strip()
            user_email = (user.get("email") or "").lower().strip()
            is_admin = (user_role == "admin" or user_email == ADMIN_EMAIL)

            if not is_admin:
                app.logger.warning(
                    "Security: Unauthorized administrative access attempt by user_id=%s, email=%s on path=%s",
                    user_id, user_email, request.path
                )
                return render_template("403.html"), 403

            session["user_role"] = "admin"
            session["is_admin"] = True

        except Exception as error:
            app.logger.error("Admin validation error: %s", error)
            return render_template("403.html"), 403

        return function(*args, **kwargs)

    return wrapper


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if session.get(
        "user_id"
    ):

        return redirect(
            url_for(
                "dashboard"
            )
        )

    return redirect(
        url_for(
            "login"
        )
    )


# ============================================================
# LOGIN PAGE (Email/Password & Google OAuth)
# ============================================================

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "").strip()

        if not email or not password:
            return render_template(
                "login.html",
                error="Please enter both your email address and password."
            )

        if not is_valid_email(email):
            return render_template(
                "login.html",
                error="Please enter a valid email address (e.g. yourname@gmail.com)."
            )

        connection = get_db_connection()
        try:
            user = connection.execute(
                """
                SELECT id, name, email, password, google_id, role
                FROM users
                WHERE LOWER(email) = %s
                """,
                (email,)
            ).fetchone()

            if user:
                user_role = (user.get("role") or ("admin" if email == ADMIN_EMAIL else "user")).lower().strip()
                is_admin = (user_role == "admin" or email == ADMIN_EMAIL)

                # Existing user: check password
                if check_password_hash(user["password"], password):
                    session.clear()
                    session["user_id"] = user["id"]
                    session["user_name"] = user.get("name") or email.split("@")[0]
                    session["user_email"] = user.get("email")
                    session["user_role"] = "admin" if is_admin else "user"
                    session["is_admin"] = is_admin
                    session["auth_type"] = "password"

                    app.logger.info("User logged in via email: %s (id=%s, role=%s)", email, user["id"], session["user_role"])
                    return redirect(url_for("dashboard"))
                elif user.get("google_id"):
                    # User originally signed in with Google OAuth and is now setting a password
                    hashed_pwd = generate_password_hash(password)
                    connection.execute(
                        """
                        UPDATE users
                        SET password = %s
                        WHERE id = %s
                        """,
                        (hashed_pwd, user["id"])
                    )
                    connection.commit()
                    session.clear()
                    session["user_id"] = user["id"]
                    session["user_name"] = user.get("name") or email.split("@")[0]
                    session["user_email"] = user.get("email")
                    session["user_role"] = "admin" if is_admin else "user"
                    session["is_admin"] = is_admin
                    session["auth_type"] = "password"

                    app.logger.info("Google user set password and logged in: %s (id=%s, role=%s)", email, user["id"], session["user_role"])
                    return redirect(url_for("dashboard"))
                else:
                    return render_template(
                        "login.html",
                        error="Incorrect password. Please verify your password and try again."
                    )
            else:
                # New user: automatically register with real email and password
                hashed_pwd = generate_password_hash(password)
                display_name = email.split("@")[0].replace(".", " ").replace("_", " ").title()
                new_role = "admin" if email == ADMIN_EMAIL else "user"
                cursor = connection.execute(
                    """
                    INSERT INTO users (name, email, password, role)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, name, email, role
                    """,
                    (display_name, email, hashed_pwd, new_role)
                )
                connection.commit()
                new_user = cursor.fetchone()

                session.clear()
                session["user_id"] = new_user["id"]
                session["user_name"] = new_user.get("name") or display_name
                session["user_email"] = new_user.get("email") or email
                session["user_role"] = new_role
                session["is_admin"] = (new_role == "admin")
                session["auth_type"] = "password"

                app.logger.info("New user auto-registered and logged in: %s (id=%s, role=%s)", email, new_user["id"], new_role)
                return redirect(url_for("dashboard"))

        except Exception as e:
            app.logger.exception("Login database error: %s", e)
            return render_template(
                "login.html",
                error="An error occurred while logging in. Please try again."
            )
        finally:
            connection.close()

    return render_template("login.html")


# ============================================================
# GOOGLE LOGIN
# ============================================================

@app.route("/google/login")
def google_login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    try:
        login_hint = (request.args.get("login_hint") or "").strip()
        extra_args = {}
        if login_hint and is_valid_email(login_hint):
            extra_args["login_hint"] = login_hint
        return google.authorize_redirect(GOOGLE_REDIRECT_URI, **extra_args)
    except Exception as error:
        app.logger.exception(
            "Google OAuth start failed: %s",
            error
        )
        return render_template(
            "login.html",
            error=(
                "Unable to start Google login. "
                "Please try again."
            )
        )


# ============================================================
# GOOGLE OAUTH CALLBACK
# ============================================================

@app.route("/google/callback")
def google_callback():

    try:

        token = google.authorize_access_token()

        userinfo = token.get("userinfo")

        if not userinfo:
            userinfo = google.userinfo()

        if not userinfo:
            return render_template(
                "login.html",
                error="Google could not verify your account."
            )

        # ----------------------------------------------------
        # GET GOOGLE USER INFORMATION
        # ----------------------------------------------------

        google_id = userinfo.get("sub")

        email = (
            userinfo.get("email") or ""
        ).strip().lower()

        name = (
            userinfo.get("name") or ""
        ).strip()

        picture = (
            userinfo.get("picture") or ""
        )

        email_verified = userinfo.get(
            "email_verified",
            False
        )

        # ----------------------------------------------------
        # VALIDATE GOOGLE INFORMATION
        # ----------------------------------------------------

        if not google_id:

            return render_template(
                "login.html",
                error="Google account verification failed."
            )

        if not email:

            return render_template(
                "login.html",
                error="Google did not provide an email address."
            )

        if not email_verified:

            return render_template(
                "login.html",
                error="Your Google email address is not verified."
            )

        if not name:

            name = email.split("@")[0]

        # ----------------------------------------------------
        # DATABASE CONNECTION
        # ----------------------------------------------------

        connection = get_db_connection()

        try:

            # ------------------------------------------------
            # CHECK GOOGLE ID
            # ------------------------------------------------

            user = connection.execute(
                """
                SELECT *
                FROM users
                WHERE google_id = %s
                """,
                (google_id,)
            ).fetchone()

            # ------------------------------------------------
            # IF GOOGLE ID NOT FOUND,
            # CHECK EMAIL
            # ------------------------------------------------

            if user is None:

                user = connection.execute(
                    """
                    SELECT *
                    FROM users
                    WHERE LOWER(email) = %s
                    """,
                    (email,)
                ).fetchone()

            # ------------------------------------------------
            # EXISTING USER
            # ------------------------------------------------

            if user is not None:

                connection.execute(
                    """
                    UPDATE users
                    SET
                        google_id = %s,
                        name = %s,
                        email = %s
                    WHERE id = %s
                    """,
                    (
                        google_id,
                        name,
                        email,
                        user["id"]
                    )
                )

                user_id = user["id"]

            # ------------------------------------------------
            # NEW GOOGLE USER
            # ------------------------------------------------

            else:

                unusable_password = (
                    generate_password_hash(
                        secrets.token_urlsafe(48)
                    )
                )

                cursor = connection.execute(
                    """
                    INSERT INTO users
                    (
                        name,
                        email,
                        password,
                        google_id
                    )
                    VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    RETURNING id
                    """,
                    (
                        name,
                        email,
                        unusable_password,
                        google_id
                    )
                )

                user_id = cursor.fetchone()["id"]

            # ------------------------------------------------
            # SAVE DATABASE CHANGES
            # ------------------------------------------------

            connection.commit()

            # ------------------------------------------------
            # GET USER AGAIN
            # ------------------------------------------------

            user = connection.execute(
                """
                SELECT *
                FROM users
                WHERE id = %s
                """,
                (user_id,)
            ).fetchone()

        except Exception:

            connection.rollback()

            raise

        finally:

            connection.close()

        # ----------------------------------------------------
        # CHECK USER
        # ----------------------------------------------------

        if user is None:

            return render_template(
                "login.html",
                error=(
                    "Unable to create your "
                    "College Predictor account."
                )
            )

        # ----------------------------------------------------
        # CREATE LOGIN SESSION
        # ----------------------------------------------------

        session.clear()

        session["user_id"] = user["id"]

        session["user_name"] = user["name"]

        session["user_email"] = user["email"]

        user_role = (user.get("role") or ("admin" if (user.get("email") or "").lower() == ADMIN_EMAIL else "user")).lower().strip()
        is_admin = (user_role == "admin" or (user.get("email") or "").lower() == ADMIN_EMAIL)
        session["user_role"] = "admin" if is_admin else "user"
        session["is_admin"] = is_admin

        session["google_id"] = google_id

        session["google_picture"] = picture

        session["login_method"] = "google"

        # ----------------------------------------------------
        # LOG SUCCESS
        # ----------------------------------------------------

        app.logger.info(
            "Google login successful | User=%s | Email=%s",
            user["id"],
            user["email"]
        )

        # ----------------------------------------------------
        # REDIRECT TO DASHBOARD
        # ----------------------------------------------------

        return redirect(
            url_for("dashboard")
        )

    except Exception as error:

        app.logger.exception(
            "Google OAuth callback failed: %s",
            error
        )

        return render_template(
            "login.html",
            error="Google login failed. Please try again."
        )

# ============================================================
# LEGACY SIGNUP URL
# ============================================================
# Google is the only authentication method. Keep this redirect
# so an old signup link cannot open the previous password/OTP form.

@app.route("/signup", methods=["GET", "POST"])
@limiter.limit("3 per minute", methods=["POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "").strip()
        confirm_password = (request.form.get("confirm_password") or "").strip()

        if not name or not email or not password:
            return render_template(
                "signup.html",
                error="Please fill in all required fields."
            )

        if not is_valid_email(email):
            return render_template(
                "signup.html",
                error="Please enter a valid email address."
            )

        if len(password) < 6:
            return render_template(
                "signup.html",
                error="Password must contain at least 6 characters."
            )

        if password != confirm_password:
            return render_template(
                "signup.html",
                error="Passwords do not match."
            )

        connection = get_db_connection()
        try:
            existing = connection.execute(
                "SELECT id FROM users WHERE LOWER(email) = %s",
                (email,)
            ).fetchone()

            if existing:
                return render_template(
                    "signup.html",
                    error="An account with this email already exists. Please sign in."
                )

            hashed_password = generate_password_hash(password)
            new_role = "admin" if email == ADMIN_EMAIL else "user"

            cursor = connection.execute(
                """
                INSERT INTO users (name, email, password, role)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, email, hashed_password, new_role)
            )
            connection.commit()
            new_user = cursor.fetchone()
            user_id = new_user["id"]

            session.clear()
            session["user_id"] = user_id
            session["user_name"] = name
            session["user_email"] = email
            session["user_role"] = new_role
            session["is_admin"] = (new_role == "admin")
            session["auth_type"] = "password"

            app.logger.info("New user registered: %s (id=%s, role=%s)", email, user_id, new_role)
            return redirect(url_for("dashboard"))

        except Exception as e:
            app.logger.exception("Signup database error: %s", e)
            return render_template(
                "signup.html",
                error="An error occurred while creating your account. Please try again."
            )
        finally:
            connection.close()

    return render_template("signup.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    return redirect(url_for("login"))


# LOGOUT
# ============================================================

@app.route(
    "/logout",
    methods=["POST"]
)
@login_required
def logout():

    user_id = session.get(
        "user_id"
    )

    session.clear()

    app.logger.info(
        "User logged out | User=%s",
        user_id
    )

    response = redirect(
        url_for(
            "login"
        )
    )

    response.headers[
        "Clear-Site-Data"
    ] = '"cache"'

    return response

# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    # --------------------------------------------------------
    # STEP 1: GET LOGGED-IN USER ID
    # --------------------------------------------------------

    user_id = session.get("user_id")

    connection = get_db_connection()

    try:

        # ----------------------------------------------------
        # STEP 2: GET USER DETAILS INCLUDING PROFILE IMAGE
        # ----------------------------------------------------

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        ).fetchone()

        # ----------------------------------------------------
        # STEP 3: GET PREDICTION HISTORY
        # ----------------------------------------------------

        history = connection.execute(
            """
            SELECT *
            FROM prediction_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        ).fetchall()

    finally:
        connection.close()

    # --------------------------------------------------------
    # STEP 4: CHECK USER
    # --------------------------------------------------------

    if user is None:
        session.clear()
        return redirect(url_for("login"))

    # --------------------------------------------------------
    # STEP 5: GET STUDENT NAME
    # --------------------------------------------------------

    user_name = (
        user.get("name")
        or session.get("user_name")
        or "Student"
    )

    session["user_name"] = user_name

    # --------------------------------------------------------
    # STEP 6: GET PROFILE IMAGE
    # --------------------------------------------------------

    profile_image = (
        user.get("profile_image")
        or session.get("profile_image")
    )

    google_picture = session.get("google_picture")

    # Keep the session synchronized with the database.
    if profile_image:
        session["profile_image"] = profile_image
    else:
        session.pop("profile_image", None)

    # --------------------------------------------------------
    # STEP 7: CALCULATE PREDICTION STATISTICS
    # --------------------------------------------------------

    total_predictions = len(history)

    kcet_predictions = sum(
        1
        for item in history
        if (item.get("exam") or "").upper() == "KCET"
    )

    eamcet_predictions = sum(
        1
        for item in history
        if (item.get("exam") or "").upper() == "EAMCET"
    )

    recent_predictions = history[:5]

    # --------------------------------------------------------
    # STEP 8: LOAD DASHBOARD
    # --------------------------------------------------------

    return render_template(
        "dashboard.html",

        # Student profile
        user=user,
        user_name=user_name,
        profile_image=profile_image,
        google_picture=google_picture,

        # College statistics
        total_colleges=len(kcet_df) + len(eamcet_df),
        kcet_colleges=len(kcet_df),
        eamcet_colleges=len(eamcet_df),

        # Prediction statistics
        total_predictions=total_predictions,
        kcet_predictions=kcet_predictions,
        eamcet_predictions=eamcet_predictions,
        recent_predictions=recent_predictions,

        # Latest prediction details
        latest_prediction=history[0] if history else None,
        latest_exam=history[0].get("exam") if history else None,
        latest_rank=history[0].get("rank") if history else None,
        latest_branch=history[0].get("branch") if history else None,

        # Popular colleges
        featured_colleges=[]
    )

# ============================================================
# ADMIN CONTACT PAGE
# ============================================================

@app.route("/admin-contact")
@login_required
def admin_contact():

    return render_template("admin.html")

@app.route("/admin/dashboard")
@app.route("/admin_dashboard")
def redirect_admin_dashboard():
    return redirect(url_for("dashboard"))

# ============================================================
# COLLEGES
# ============================================================

@app.route("/colleges")
@login_required
def colleges():

    exam = (
        request.args.get(
            "exam",
            "all"
        )
        .strip()
        .lower()
    )

    # ========================================================
    # SELECT DATASET
    # ========================================================

    if exam == "kcet":

        dataframe = kcet_df.copy()

        selected_exam = "KCET"

    elif exam == "eamcet":

        dataframe = eamcet_df.copy()

        selected_exam = "EAMCET"

    else:

        dataframe = pd.concat(
            [
                kcet_df,
                eamcet_df
            ],
            ignore_index=True
        )

        selected_exam = "ALL"

    # ========================================================
    # GROUP COLLEGES
    #
    # One college = one card
    # Multiple branches = inside that card
    # ========================================================

    grouped_colleges = {}

    for index, row in dataframe.iterrows():

        college_name = str(
            row.get(
                "College Name",
                ""
            )
        ).strip()

        if not college_name:
            continue

        row_exam = str(
            row.get(
                "Exam",
                selected_exam
            )
        ).strip().upper()

        # ----------------------------------------------------
        # Create unique key using exam + college name
        # ----------------------------------------------------

        group_key = (
            row_exam,
            college_name.lower()
        )

        # ----------------------------------------------------
        # Create college if not already present
        # ----------------------------------------------------

        if group_key not in grouped_colleges:

            college = row.to_dict()

            college["College Name"] = college_name

            college["Exam"] = row_exam

            # Keep an index relative to the exam-specific dataset.
            # When showing ALL exams, EAMCET rows come after KCET rows
            # in the concatenated dataframe, so remove that offset.
            if selected_exam == "ALL" and row_exam == "EAMCET":
                source_index = int(index) - len(kcet_df)
            else:
                source_index = int(index)

            college["SourceIndex"] = source_index
            college["Original Index"] = source_index

            college["Website"] = clean_website(
                row.get(
                    "Website",
                    ""
                ),
                college_name
            )

            college["Image"] = clean_image(
                row.get(
                    "Image",
                    ""
                )
            )

            # ------------------------------------------------
            # Branch list
            # ------------------------------------------------

            college["Branches"] = []

            grouped_colleges[group_key] = college

        else:

            college = grouped_colleges[group_key]

        # ====================================================
        # CREATE BRANCH INFORMATION
        # ====================================================

        branch_name = str(
            row.get(
                "Branch",
                ""
            )
        ).strip()

        if not branch_name:

            branch_name = "N/A"

        # ----------------------------------------------------
        # Management fees
        # ----------------------------------------------------

        management_fees = clean_numeric(
            row.get(
                "Management Fees",
                0
            )
        )

        # ----------------------------------------------------
        # Exam fees
        # ----------------------------------------------------

        if row_exam == "KCET":

            exam_fees = clean_numeric(
                row.get(
                    "Kcet Fees",
                    row.get(
                        "KCET Fees",
                        0
                    )
                )
            )

        else:

            exam_fees = clean_numeric(
                row.get(
                    "Eamcet Fees",
                    row.get(
                        "EAMCET Fees",
                        0
                    )
                )
            )

        # ====================================================
        # BRANCH OBJECT
        # ====================================================

        branch_data = {

            "Branch": branch_name,

            "Cutoff Rank": clean_rank(
                row.get(
                    "Cutoff Rank",
                    0
                )
            ),

            "Management Fees": management_fees,

            "Exam Fees": exam_fees,

            "Rating": clean_rating(
                row.get(
                    "Rating",
                    0
                )
            )
        }

        # ====================================================
        # AVOID DUPLICATE BRANCHES
        # ====================================================

        existing_branches = [
            str(
                branch.get(
                    "Branch",
                    ""
                )
            ).upper()
            for branch in college[
                "Branches"
            ]
        ]

        if branch_name.upper() not in existing_branches:

            college[
                "Branches"
            ].append(
                branch_data
            )

    # ========================================================
    # FINAL COLLEGE LIST
    # ========================================================

    college_list = list(
        grouped_colleges.values()
    )

    # ========================================================
    # RETURN PAGE
    # ========================================================

    return render_template(

        "colleges.html",

        colleges=college_list,

        selected_exam=selected_exam
    )

# ============================================================
# COLLEGE DETAILS
# ============================================================

@app.route(
    "/college/<exam>/<int:index>"
)
@login_required
def college_details(
    exam,
    index
):

    exam = exam.lower().strip()

    if exam == "kcet":

        dataframe = kcet_df

        display_exam = "KCET"

    elif exam == "eamcet":

        dataframe = eamcet_df

        display_exam = "EAMCET"

    else:

        return redirect(
            url_for(
                "colleges"
            )
        )

    if not (
        0 <= index < len(dataframe)
    ):

        return redirect(
            url_for(
                "colleges"
            )
        )

    college = (
        dataframe
        .iloc[index]
        .to_dict()
    )

    college["Exam"] = display_exam

    college["SourceIndex"] = index

    college["Website"] = clean_website(

        college.get(
            "Website"
        ),

        college.get(
            "College Name",
            ""
        )
    )

    college["Image"] = clean_image(
        college.get(
            "Image",
            ""
        )
    )

    college["Exam Fees"] = (
        clean_numeric(
            college.get(
                "Kcet Fees"
                if display_exam == "KCET"
                else "Eamcet Fees",
                0
            )
        )
    )

    college["MapQuery"] = (
        str(
            college.get(
                "College Name",
                ""
            )
        )
        +
        ", "
        +
        str(
            college.get(
                "Location",
                ""
            )
        )
    )

    return render_template(

        "college_details.html",

        college=college,

        exam=display_exam,

        index=index
    )





# ============================================================
# PREDICTOR PAGE
# ============================================================

@app.route(
    "/predictor"
)
@login_required
def predictor():

    exam = request.args.get(
        "exam",
        ""
    ).strip().upper()

    rank = request.args.get(
        "rank",
        ""
    ).strip()

    branch = request.args.get(
        "branch",
        ""
    ).strip().upper()

    branch_values = sorted(

        set(

            list(
                kcet_df[
                    "Branch"
                ].dropna().unique()
            )
            +
            list(
                eamcet_df[
                    "Branch"
                ].dropna().unique()
            )

        )

    )

    branch_values = [
        branch
        for branch in branch_values
        if branch in ALLOWED_BRANCHES
    ]

    return render_template(

        "predictor.html",

        branches=branch_values,

        selected_exam=exam,

        selected_rank=rank,

        selected_branch=branch
    )


# ============================================================
# RECOMMENDATION SCORE
# ============================================================

def calculate_recommendation_score(
    rank,
    cutoff,
    ml_score,
    rating,
    fee_score
):

    rank_difference = abs(
        cutoff - rank
    )

    rank_compatibility = (
        1
        -
        (
            rank_difference
            /
            10000
        )
    )

    rank_compatibility = max(
        0,
        min(
            1,
            rank_compatibility
        )
    )

    ml_component = max(
        0,
        min(
            1,
            ml_score
        )
    )

    rating_component = max(
        0,
        min(
            1,
            rating / 5
        )
    )

    fee_component = max(
        0,
        min(
            1,
            fee_score
        )
    )

    recommendation_score = (

        rank_compatibility * 0.40

        +

        ml_component * 0.35

        +

        rating_component * 0.15

        +

        fee_component * 0.10
    )

    return round(
        recommendation_score,
        4
    )
# ============================================================
# NORMALIZE BRANCH NAME
# ============================================================

def normalize_branch(branch):

    branch = str(
        branch or ""
    ).strip().upper()

    branch_map = {

        # --------------------------------------------
        # COMPUTER SCIENCE
        # --------------------------------------------

        "CSE":
            "CSE",

        "COMPUTER SCIENCE":
            "CSE",

        "COMPUTER SCIENCE AND ENGINEERING":
            "CSE",


        # --------------------------------------------
        # ARTIFICIAL INTELLIGENCE
        # --------------------------------------------

        "AIML":
            "AIML",

        "AI ML":
            "AIML",

        "AI&ML":
            "AIML",

        "AI AND ML":
            "AIML",

        "ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING":
            "AIML",


        # --------------------------------------------
        # INFORMATION SCIENCE
        # --------------------------------------------

        "ISE":
            "ISE",

        "INFORMATION SCIENCE":
            "ISE",

        "INFORMATION SCIENCE AND ENGINEERING":
            "ISE",


        # --------------------------------------------
        # ELECTRONICS
        # --------------------------------------------

        "ECE":
            "ECE",

        "ELECTRONICS":
            "ECE",

        "ELECTRONICS AND COMMUNICATION":
            "ECE",

        "ELECTRONICS AND COMMUNICATION ENGINEERING":
            "ECE",


        # --------------------------------------------
        # MECHANICAL
        # --------------------------------------------

        "ME":
            "ME",

        "MECHANICAL":
            "ME",

        "MECHANICAL ENGINEERING":
            "ME",


        # --------------------------------------------
        # ELECTRICAL
        # --------------------------------------------

        "EEE":
            "EEE",

        "ELECTRICAL":
            "EEE",

        "ELECTRICAL AND ELECTRONICS":
            "EEE",

        "ELECTRICAL AND ELECTRONICS ENGINEERING":
            "EEE",


        # --------------------------------------------
        # CIVIL
        # --------------------------------------------

        "CIVIL":
            "CIVIL",

        "CIVIL ENGINEERING":
            "CIVIL"
    }

    return branch_map.get(
        branch,
        branch
    )

# ============================================================
# PREDICT COLLEGES
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
@limiter.limit("15 per minute")
def predict():

    # ========================================================
    # GET FORM DATA
    # ========================================================

    exam = (
        request.form.get("exam") or ""
    ).strip().lower()

    branch = normalize_branch(
        request.form.get("branch", "")
    )

    rank_text = (
        request.form.get("rank") or ""
    ).strip()

    category = (
        request.form.get("category") or ""
    ).strip()

    gender = (
        request.form.get("gender") or ""
    ).strip()

    preferred_location = (
        request.form.get("preferred_location") or ""
    ).strip()

    state = (
        request.form.get("state") or ""
    ).strip()

    fee_range = (
        request.form.get("fee_range") or ""
    ).strip()

    app.logger.info(
        "Prediction request | exam=%s | branch=%s | rank=%s",
        exam,
        branch,
        rank_text
    )

    # ========================================================
    # VALIDATE EXAM
    # ========================================================

    allowed_exams = {
        str(value).strip().lower()
        for value in ALLOWED_EXAMS
    }

    if exam not in allowed_exams:

        flash(
            "Please select a valid entrance exam.",
            "error"
        )

        return redirect(url_for("predictor"))

    # ========================================================
    # VALIDATE RANK
    # ========================================================

    try:
        rank = int(float(rank_text))

    except (ValueError, TypeError, OverflowError):

        app.logger.warning(
            "Invalid rank received | rank=%s",
            rank_text
        )

        flash(
            "Please enter a valid rank.",
            "error"
        )

        return redirect(url_for("predictor"))

    if rank <= 0:

        flash(
            "Rank must be greater than 0.",
            "error"
        )

        return redirect(url_for("predictor"))

    # ========================================================
    # VALIDATE BRANCH
    # ========================================================

    if not branch:

        flash(
            "Please select a valid branch.",
            "error"
        )

        return redirect(url_for("predictor"))

    # ========================================================
    # SELECT DATASET
    # ========================================================

    if exam == "kcet":

        dataframe = kcet_df.copy()
        exam_name = "KCET"

    elif exam == "eamcet":

        dataframe = eamcet_df.copy()
        exam_name = "EAMCET"

    else:

        flash(
            "Invalid entrance exam.",
            "error"
        )

        return redirect(url_for("predictor"))

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if dataframe.empty:

        app.logger.warning(
            "Dataset is empty | exam=%s",
            exam
        )

        flash(
            "No college data is available for this exam.",
            "error"
        )

        return redirect(url_for("predictor"))

    # ========================================================
    # NORMALIZE DATASET BRANCHES
    # ========================================================

    dataframe = dataframe.copy()

    if "Branch" not in dataframe.columns:

        app.logger.error(
            "Branch column missing from %s dataset.",
            exam_name
        )

        flash(
            "College branch data is unavailable.",
            "error"
        )

        return redirect(url_for("predictor"))

    dataframe["_NormalizedBranch"] = (
        dataframe["Branch"].apply(normalize_branch)
    )

    # ========================================================
    # FIND SELECTED BRANCH
    # ========================================================

    branch_df = dataframe[
        dataframe["_NormalizedBranch"] == branch
    ].copy()

    app.logger.info(
        "Prediction branch search | exam=%s | branch=%s | matching_rows=%s",
        exam,
        branch,
        len(branch_df)
    )

    if branch_df.empty:

        app.logger.warning(
            "No colleges found for branch | exam=%s | branch=%s",
            exam,
            branch
        )

        flash(
            (
                "No colleges were found for "
                f"{branch} in {exam_name}. "
                "Please try another branch."
            ),
            "error"
        )

        return redirect(url_for("predictor"))

    # ========================================================
    # CHECK CUTOFF COLUMN
    # ========================================================

    if "Cutoff Rank" not in branch_df.columns:

        app.logger.error(
            "Cutoff Rank column missing from %s dataset.",
            exam_name
        )

        flash(
            "College cutoff data is unavailable.",
            "error"
        )

        return redirect(url_for("predictor"))

    # ========================================================
    # RANK FILTER: STUDENT RANK ± 3000
    # ========================================================

    min_rank = max(1, rank - 5000)
    max_rank = rank + 5000

    branch_df["_CleanCutoff"] = (
        branch_df["Cutoff Rank"].apply(clean_rank)
    )

    rank_filtered = branch_df[
        (branch_df["_CleanCutoff"] >= min_rank)
        & (branch_df["_CleanCutoff"] <= max_rank)
        & (branch_df["_CleanCutoff"] > 0)
    ].copy()

    app.logger.info(
        "Rank filter applied | student_rank=%s | min_rank=%s | max_rank=%s | matching_colleges=%s",
        rank,
        min_rank,
        max_rank,
        len(rank_filtered)
    )

    # IMPORTANT:
    # Do not fall back to all branch colleges.
    # Only colleges within the requested cutoff range
    # are allowed into the prediction results.

    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    results = []

    for index, row in rank_filtered.iterrows():

        # ====================================================
        # BASIC COLLEGE DATA
        # ====================================================

        cutoff = clean_rank(
            row.get("Cutoff Rank", 0)
        )

        rating = clean_rating(
            row.get("Rating", 0)
        )

        management_fees = clean_numeric(
            row.get("Management Fees", 0)
        )

        hostel_fees = clean_numeric(
            row.get("Hostel Fees", 0)
        )

        # ====================================================
        # EXAM-SPECIFIC FEE
        # ====================================================

        if exam == "eamcet":

            exam_fee = clean_numeric(
                row.get(
                    "Eamcet Fees",
                    row.get("EAMCET Fees", 0)
                )
            )

            exam_fee_label = "EAMCET Fee"

        else:

            exam_fee = clean_numeric(
                row.get(
                    "Kcet Fees",
                    row.get("KCET Fees", 0)
                )
            )

            exam_fee_label = "KCET Fee"

        # ====================================================
        # RANK COMPATIBILITY SCORE
        # ====================================================

        if cutoff > 0:

            rank_difference = abs(cutoff - rank)

            rank_score = (
                1.0
                - (
                    rank_difference
                    / max(cutoff, rank, 1)
                )
            )

        else:

            rank_score = 0.5

        rank_score = max(
            0.0,
            min(1.0, float(rank_score))
        )

        # ====================================================
        # ML PREDICTION
        # ====================================================

        ml_score = None

        if model is not None:

            try:

                feature_row = pd.DataFrame(
                    [
                        {
                            "Exam": exam.upper(),
                            "Branch": branch,
                            "Student Rank": rank,
                            "Rating": rating,
                            "Management Fees": management_fees,
                            "Hostel Fees": hostel_fees,
                            "Cutoff Rank": cutoff
                        }
                    ]
                )

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        feature_row
                    )

                    if len(probabilities) > 0:

                        probability_row = probabilities[0]

                        if len(probability_row) >= 2:

                            ml_score = float(
                                probability_row[1]
                            )

                        else:

                            ml_score = float(
                                probability_row[0]
                            )

                elif hasattr(model, "predict"):

                    prediction_value = model.predict(
                        feature_row
                    )[0]

                    ml_score = float(prediction_value)

            except Exception as model_error:

                app.logger.warning(
                    "ML prediction fallback used | exam=%s | branch=%s | college=%s | error=%s",
                    exam,
                    branch,
                    row.get("College Name", ""),
                    model_error
                )

                ml_score = None

        # ====================================================
        # FALLBACK ML SCORE
        # ====================================================

        if ml_score is None:

            if cutoff > 0:

                ml_score = (
                    1.0
                    - (
                        abs(cutoff - rank)
                        / max(cutoff, rank, 1)
                    )
                )

            else:

                ml_score = 0.5

        ml_score = max(
            0.0,
            min(1.0, float(ml_score))
        )

        # ====================================================
        # RATING SCORE
        # ====================================================

        if rating > 0:

            rating_score = min(
                1.0,
                float(rating) / 5.0
            )

        else:

            rating_score = 0.5

        # ====================================================
        # FEE SCORE
        # ====================================================

        if management_fees <= 0:

            fee_score = 0.5

        else:

            fee_score = max(
                0.0,
                min(
                    1.0,
                    1.0 - (management_fees / 1000000.0)
                )
            )

        # ====================================================
        # FINAL SCORE
        #
        # Rank:   40%
        # ML:     35%
        # Rating: 15%
        # Fees:   10%
        # ====================================================

        final_score = (
            (rank_score * 0.40)
            + (ml_score * 0.35)
            + (rating_score * 0.15)
            + (fee_score * 0.10)
        )

        final_score = max(
            0.0,
            min(1.0, float(final_score))
        )

        chance_percentage = round(
            final_score * 100,
            1
        )

        # ====================================================
        # PREDICTION LABEL
        # ====================================================

        if chance_percentage >= 75:

            prediction_label = "High Chance"

        elif chance_percentage >= 50:

            prediction_label = "Good Chance"

        elif chance_percentage >= 30:

            prediction_label = "Moderate Chance"

        else:

            prediction_label = "Low Chance"

        # ====================================================
        # CREATE RESULT
        # ====================================================

        result = row.to_dict()

        result["Exam"] = exam_name
        result["Branch"] = branch
        result["Student Rank"] = rank
        result["Cutoff Rank"] = cutoff
        result["Rating"] = rating
        result["Management Fees"] = management_fees
        result["Hostel Fees"] = hostel_fees

        result["Exam Fee"] = exam_fee
        result["Exam Fee Label"] = exam_fee_label

        result["ML Score"] = round(ml_score * 100, 1)
        result["Rank Score"] = round(rank_score * 100, 1)
        result["Fee Score"] = round(fee_score * 100, 1)

        result["Chance"] = chance_percentage
        result["Chance Percentage"] = chance_percentage
        result["Prediction"] = prediction_label

        result["SourceIndex"] = int(index)

        result["Website"] = clean_website(
            row.get("Website", ""),
            row.get("College Name", "")
        )

        result["Image"] = get_result_image(
            row.get("Image", "")
        )

        results.append(result)

    # ========================================================
    # SORT RESULTS BY CHANCE
    # ========================================================

    results.sort(
        key=lambda item: item.get("Chance", 0),
        reverse=True
    )

    # ========================================================
    # LIMIT RESULTS
    # ========================================================

    results = results[:20]

    # ========================================================
    # SAVE PREDICTION HISTORY
    # ========================================================

    connection = get_db_connection()

    try:

        connection.execute(
            """
            INSERT INTO prediction_history
            (
                user_id,
                exam,
                rank,
                branch,
                category,
                gender,
                preferred_location,
                state,
                fee_range
            )
            VALUES
            (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            """,
            (
                session.get("user_id"),
                exam,
                rank,
                branch,
                category,
                gender,
                preferred_location,
                state,
                fee_range
            )
        )

        connection.commit()

    except Exception as history_error:

        connection.rollback()

        app.logger.warning(
            "Prediction history save failed: %s",
            history_error
        )

    finally:

        connection.close()

    # ========================================================
    # HANDLE NO MATCHING COLLEGES
    # ========================================================

    if not results:

        app.logger.info(
            "No matching colleges | exam=%s | branch=%s | rank=%s | min_rank=%s | max_rank=%s",
            exam,
            branch,
            rank,
            min_rank,
            max_rank
        )

        flash(
            (
                f"No colleges with valid cutoff ranks between "
                f"{min_rank} and {max_rank} were found for "
                f"{branch} in {exam_name}. "
                "Please try another rank or branch."
            ),
            "error"
        )

        return redirect(url_for("predictor"))

    # ========================================================
    # FINAL LOG
    # ========================================================

    app.logger.info(
        "Prediction completed | exam=%s | branch=%s | rank=%s | results=%s",
        exam,
        branch,
        rank,
        len(results)
    )

    # ========================================================
    # RENDER RESULT PAGE
    # ========================================================

    return render_template(
        "result.html",
        colleges=results,
        results=results,
        exam=exam,
        exam_name=exam_name,
        rank=rank,
        branch=branch,
        category=category,
        gender=gender,
        preferred_location=preferred_location,
        state=state,
        fee_range=fee_range,
        min_rank=min_rank,
        max_rank=max_rank
    )

# ============================================================
# PROFILE / HELP / ABOUT / HISTORY
# ============================================================


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        return update_profile()

    connection = get_db_connection()

    try:
        user = connection.execute(
            """
            SELECT
                id,
                name,
                email,
                phone,
                profile_image,
                created_at
            FROM users
            WHERE id = %s
            """,
            (session.get("user_id"),)
        ).fetchone()

    finally:
        connection.close()

    if user is None:
        session.clear()
        return redirect(url_for("login"))

    user = dict(user)

    # Prefer the student's uploaded image.
    # Otherwise, use their Google profile picture.
    user["profile_picture"] = (
        user.get("profile_image")
        or session.get("google_picture")
    )

    user["google_id"] = session.get("google_id")
    user["login_method"] = session.get("login_method")

    app.logger.info(
        "Profile loaded | User=%s | Uploaded picture=%s",
        user["id"],
        bool(user.get("profile_image"))
    )

    return render_template(
        "profile.html",
        user=user,
        profile_image=user.get("profile_picture") or user.get("profile_image"),
        google_picture=session.get("google_picture"),
        error=None
    )


# ============================================================
# UPDATE PROFILE
# ============================================================

@app.route("/update-profile", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def update_profile():

    name = request.form.get("name", "").strip()[:100]
    phone = request.form.get("phone", "").strip()

    if not name:
        return render_template(
            "profile.html",
            user={
                "name": name,
                "phone": phone,
                "email": session.get("user_email", ""),
                "profile_picture": session.get("google_picture")
            },
            error="Name cannot be empty."
        ), 400

    # Allow common international phone-number formats.
    if phone and not re.fullmatch(r"[0-9+() .-]{7,20}", phone):
        return redirect(url_for("profile", error="invalid_phone"))

    uploaded_file = request.files.get("profile_image")
    saved_filename = None
    final_image_path = None

    if uploaded_file and uploaded_file.filename:
        original_name = secure_filename(uploaded_file.filename)

        if (
            "." not in original_name
            or original_name.rsplit(".", 1)[1].lower()
            not in ALLOWED_PROFILE_IMAGE_EXTENSIONS
        ):
            return redirect(url_for("profile", error="invalid_image"))

        # Deep inspection: Verify binary magic bytes to block disguised executable scripts
        header = uploaded_file.read(16)
        uploaded_file.seek(0)
        is_valid_magic = (
            header.startswith(b"\x89PNG\r\n\x1a\n") or
            header.startswith(b"\xff\xd8\xff") or
            (header.startswith(b"RIFF") and b"WEBP" in header)
        )
        if not is_valid_magic:
            app.logger.warning("Security Shield: Rejected file upload with invalid magic bytes: %s", original_name)
            return redirect(url_for("profile", error="invalid_image"))

        extension = original_name.rsplit(".", 1)[1].lower()

        # Use a random filename to prevent collisions and
        # avoid trusting the user's original filename.
        saved_filename = (
            secrets.token_hex(16) + "." + extension
        )

        final_image_path = os.path.join(
            PROFILE_UPLOAD_FOLDER,
            saved_filename
        )

        try:
            uploaded_file.save(final_image_path)

        except Exception:
            app.logger.exception("Profile image upload failed.")
            return redirect(url_for("profile", error="upload_failed"))

    connection = get_db_connection()

    old_image = None

    try:
        current_user = connection.execute(
            """
            SELECT profile_image
            FROM users
            WHERE id = %s
            """,
            (session.get("user_id"),)
        ).fetchone()

        if current_user is None:
            if final_image_path and os.path.exists(final_image_path):
                os.remove(final_image_path)

            session.clear()
            return redirect(url_for("login"))

        old_image = current_user.get("profile_image")

        if saved_filename:
            connection.execute(
                """
                UPDATE users
                SET name = %s,
                    phone = %s,
                    profile_image = %s
                WHERE id = %s
                """,
                (
                    name,
                    phone or None,
                    url_for(
                        "static",
                        filename="uploads/profiles/" + saved_filename
                    ),
                    session.get("user_id")
                )
            )
        else:
            connection.execute(
                """
                UPDATE users
                SET name = %s,
                    phone = %s
                WHERE id = %s
                """,
                (
                    name,
                    phone or None,
                    session.get("user_id")
                )
            )

        connection.commit()

        session["user_name"] = name
        if saved_filename:
            session["profile_image"] = url_for("static", filename="uploads/profiles/" + saved_filename)

        flash("Profile updated successfully.", "success")

    except Exception:
        connection.rollback()

        if final_image_path and os.path.exists(final_image_path):
            os.remove(final_image_path)

        app.logger.exception("Profile update failed.")

        flash("Unable to update your profile.", "error")

    finally:
        connection.close()

    # Remove the previous locally uploaded image only after
    # the database successfully points to the new image.
    if (
        saved_filename
        and old_image
        and old_image.startswith("/static/uploads/profiles/")
    ):
        old_filename = os.path.basename(old_image)
        old_path = os.path.join(
            PROFILE_UPLOAD_FOLDER,
            old_filename
        )

        if os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except OSError:
                app.logger.warning(
                    "Could not remove previous profile image."
                )

    return redirect(url_for("profile", updated="1"))


# ============================================================
# COLLEGE IMAGE PATH HELPER
# ============================================================

def get_result_image(image_value):

    # --------------------------------------------------------
    # No image value
    # --------------------------------------------------------

    if image_value is None:
        return "images/default.jpg"


    # --------------------------------------------------------
    # Convert image value to string
    # --------------------------------------------------------

    image = str(
        image_value
    ).strip()


    # --------------------------------------------------------
    # Empty image value
    # --------------------------------------------------------

    if not image:
        return "images/default.jpg"


    # --------------------------------------------------------
    # External image URL
    # --------------------------------------------------------

    if (
        image.startswith("http://")
        or image.startswith("https://")
    ):

        return image


    # --------------------------------------------------------
    # Remove leading slash
    # Example:
    # /images/college.jpg
    # becomes:
    # images/college.jpg
    # --------------------------------------------------------

    image = image.lstrip("/")


    # --------------------------------------------------------
    # Remove "static/" if it already exists
    #
    # Example:
    # static/images/college.jpg
    #
    # becomes:
    # images/college.jpg
    # --------------------------------------------------------

    if image.lower().startswith("static/"):

        image = image[7:]


    # --------------------------------------------------------
    # Add images/ folder if it is missing
    #
    # Example:
    # Reva University.jpg
    #
    # becomes:
    # images/Reva University.jpg
    # --------------------------------------------------------

    if not image.lower().startswith("images/"):

        image = "images/" + image


    # --------------------------------------------------------
    # Return final image path
    # --------------------------------------------------------

    return image

@app.route("/help")
@login_required
def help_page(): return render_template("help.html")

@app.route("/about")
@login_required
def about(): return render_template("about.html")

@app.route("/prediction-history")
@login_required
def prediction_history():
    connection=get_db_connection(); history=connection.execute("SELECT * FROM prediction_history WHERE user_id=%s ORDER BY created_at DESC",(session.get("user_id"),)).fetchall(); connection.close()
    return render_template("prediction_history.html",history=history)

@app.route("/delete-prediction/<int:prediction_id>",methods=["POST"])
@login_required
def delete_prediction(prediction_id):
    connection=get_db_connection()
    try: connection.execute("DELETE FROM prediction_history WHERE id=%s AND user_id=%s",(prediction_id,session.get("user_id"))); connection.commit()
    finally: connection.close()
    return redirect(url_for("prediction_history"))


# ============================================================
# SAVED COLLEGES / COMPARE
# ============================================================

def get_college_from_source_index(exam, source_index, selected_branch=None):
    """Return the selected college and its specific branch details."""

    exam = str(exam or "").strip().upper()

    dataframe = (
        kcet_df if exam == "KCET"
        else eamcet_df if exam == "EAMCET"
        else None
    )

    if dataframe is None:
        return None

    try:
        source_index = int(source_index)
    except (TypeError, ValueError):
        return None

    if not (0 <= source_index < len(dataframe)):
        return None

    # Find the original college record.
    original_row = dataframe.iloc[source_index].to_dict()
    college_name = str(original_row.get("College Name", "")).strip()

    # If a branch was selected, find that college's matching branch row.
    if selected_branch:
        selected_branch = normalize_branch(selected_branch)

        matching_rows = dataframe[
            dataframe["College Name"].astype(str).str.strip().str.casefold()
            == college_name.casefold()
        ]

        matching_rows = matching_rows[
            matching_rows["Branch"].apply(normalize_branch)
            == selected_branch
        ]

        if matching_rows.empty:
            return None

        # Use the matching branch record, not the first branch record.
        college = matching_rows.iloc[0].to_dict()
        college["Selected Branch"] = selected_branch

    else:
        # Backward compatibility for old comparison URLs.
        college = original_row
        college["Selected Branch"] = normalize_branch(
            college.get("Branch", "")
        )

    college["Exam"] = exam
    college["SourceIndex"] = source_index
    college["Original Index"] = source_index

    # Keep the branch value consistent with the branch being compared.
    college["Branch"] = college["Selected Branch"]

    college["Rating"] = clean_rating(college.get("Rating", 0))

    if exam == "KCET":
        college["Exam Fees"] = clean_numeric(
            college.get("Kcet Fees", college.get("KCET Fees", 0))
        )
    else:
        college["Exam Fees"] = clean_numeric(
            college.get("Eamcet Fees", college.get("EAMCET Fees", 0))
        )

    college["Management Fees"] = clean_numeric(
        college.get("Management Fees", 0)
    )

    college["Website"] = clean_website(
        college.get("Website", ""),
        college_name
    )
    college["Image"] = clean_image(college.get("Image", ""))

    # Fetch all branches offered by this college
    all_matching_rows = dataframe[
        dataframe["College Name"].astype(str).str.strip().str.casefold()
        == college_name.casefold()
    ]

    branches = []
    seen_branch_names = set()
    for _, b_row in all_matching_rows.iterrows():
        b_name = normalize_branch(b_row.get("Branch", ""))
        if not b_name or b_name in seen_branch_names:
            continue
        seen_branch_names.add(b_name)

        if exam == "KCET":
            b_exam_fees = clean_numeric(
                b_row.get("Kcet Fees", b_row.get("KCET Fees", 0))
            )
        else:
            b_exam_fees = clean_numeric(
                b_row.get("Eamcet Fees", b_row.get("EAMCET Fees", 0))
            )

        b_mgmt_fees = clean_numeric(b_row.get("Management Fees", 0))

        branches.append({
            "Branch": b_name,
            "Cutoff Rank": clean_rank(b_row.get("Cutoff Rank", 0)),
            "Exam Fees": b_exam_fees,
            "Management Fees": b_mgmt_fees,
            "Rating": clean_rating(b_row.get("Rating", 0)),
        })

    college["Branches"] = branches
    college["BranchMap"] = {b["Branch"]: b for b in branches}
    college["OfferedBranches"] = [b["Branch"] for b in branches]

    return college


# ============================================================
# SAVE COLLEGE
# ============================================================

@app.route(
    "/save_college",
    methods=["POST"]
)
@login_required
def save_college():

    # --------------------------------------------------------
    # GET EXAM
    # --------------------------------------------------------

    exam = (
        request.form.get("exam") or ""
    ).strip().lower()


    # --------------------------------------------------------
    # GET COLLEGE INDEX
    # --------------------------------------------------------

    index_text = (
        request.form.get("index") or ""
    ).strip()


    app.logger.info(
        "Save request received | exam=%s | index=%s | user=%s",
        exam,
        index_text,
        session.get("user_id")
    )


    # --------------------------------------------------------
    # VALIDATE EXAM
    #
    # Accept both:
    # KCET / kcet
    # EAMCET / eamcet
    # --------------------------------------------------------

    allowed_exams = {
        str(value).strip().lower()
        for value in ALLOWED_EXAMS
    }


    if exam not in allowed_exams:

        app.logger.warning(
            "Invalid save exam | received=%s | allowed=%s",
            exam,
            allowed_exams
        )

        flash(
            "Invalid exam.",
            "error"
        )

        return redirect(
            request.referrer
            or url_for("colleges")
        )


    # --------------------------------------------------------
    # VALIDATE INDEX
    # --------------------------------------------------------

    try:

        college_index = int(
            index_text
        )

    except (
        ValueError,
        TypeError
    ):

        app.logger.warning(
            "Invalid save college index | value=%s",
            index_text
        )

        flash(
            "Invalid college.",
            "error"
        )

        return redirect(
            request.referrer
            or url_for("colleges")
        )


    # --------------------------------------------------------
    # FIND COLLEGE
    # --------------------------------------------------------

    college = get_college_from_source_index(
        exam,
        college_index
    )


    if college is None:

        app.logger.warning(
            "College not found for save | exam=%s | index=%s",
            exam,
            college_index
        )

        flash(
            "College could not be found.",
            "error"
        )

        return redirect(
            request.referrer
            or url_for("colleges")
        )


    # --------------------------------------------------------
    # SAVE TO POSTGRESQL
    # --------------------------------------------------------

    connection = get_db_connection()


    try:

        connection.execute(
            """
            INSERT INTO saved_colleges
            (
                user_id,
                exam,
                college_index
            )
            VALUES
            (
                %s,
                %s,
                %s
            )
            ON CONFLICT
            (
                user_id,
                exam,
                college_index
            )
            DO NOTHING
            """,
            (
                session.get("user_id"),
                exam,
                college_index
            )
        )


        connection.commit()


        app.logger.info(
            "College saved successfully | user=%s | exam=%s | index=%s | name=%s",
            session.get("user_id"),
            exam,
            college_index,
            college.get(
                "College Name",
                ""
            )
        )


        flash(
            "College saved successfully.",
            "success"
        )


    except Exception as error:

        connection.rollback()


        app.logger.exception(
            "Save college failed: %s",
            error
        )


        flash(
            "Unable to save college.",
            "error"
        )


    finally:

        connection.close()


    # --------------------------------------------------------
    # GO TO SAVED COLLEGES
    # --------------------------------------------------------

    return redirect(
        url_for("saved_colleges")
    )

@app.route("/remove-saved-college", methods=["POST"])
@login_required
def remove_saved_college():
    exam = (
        request.form.get("exam")
        or request.args.get("exam")
        or ""
    ).strip().upper()

    index_text = (
        request.form.get("index")
        or request.form.get("college_index")
        or request.form.get("source_index")
        or request.form.get("original_index")
        or request.args.get("index")
        or request.args.get("college_index")
        or request.args.get("source_index")
        or request.args.get("original_index")
        or ""
    ).strip()

    if ":" in index_text:
        possible_exam, possible_index = index_text.split(":", 1)
        if possible_exam.strip().upper() in ALLOWED_EXAMS:
            exam = possible_exam.strip().upper()
            index_text = possible_index.strip()

    try:
        college_index = int(index_text)
    except (ValueError, TypeError):
        return redirect(url_for("saved_colleges"))

    if exam not in ALLOWED_EXAMS:
        return redirect(url_for("saved_colleges"))

    connection = get_db_connection()
    try:
        connection.execute(
            """
            DELETE FROM saved_colleges
            WHERE user_id = %s
              AND exam = %s
              AND college_index = %s
            """,
            (session.get("user_id"), exam, college_index),
        )
        connection.commit()
        flash("College removed from saved colleges.", "success")
    finally:
        connection.close()

    return redirect(request.referrer or url_for("saved_colleges"))


# ============================================================
# SAVED COLLEGES PAGE
# ============================================================

@app.route("/saved-colleges", endpoint="saved_colleges")
@app.route("/saved_colleges", endpoint="saved_colleges_legacy")
@login_required
def saved_colleges_page():
    """Show the current user's saved colleges."""
    user_id = session.get("user_id")
    connection = get_db_connection()

    try:
        saved_rows = connection.execute(
            """
            SELECT exam, college_index, created_at
            FROM saved_colleges
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    finally:
        connection.close()

    colleges = []
    for saved_row in saved_rows:
        exam = str(saved_row.get("exam") or "").strip().upper()
        college_index = saved_row.get("college_index")
        college = get_college_from_source_index(exam, college_index)
        if college is None:
            # Keep the page usable if a saved index no longer exists in
            # the dataset; log it for diagnosis rather than crashing.
            app.logger.warning(
                "Saved college no longer exists | user=%s | exam=%s | index=%s",
                user_id, exam, college_index,
            )
            continue

        college["Exam"] = exam
        college["SourceIndex"] = int(college_index)
        college["Original Index"] = int(college_index)
        college["Saved At"] = saved_row.get("created_at")
        college["Website"] = clean_website(
            college.get("Website", ""), college.get("College Name", "")
        )
        college["Image"] = clean_image(college.get("Image", ""))
        colleges.append(college)

    return render_template(
        "saved_colleges.html",
        colleges=colleges,
        saved_colleges=colleges,
    )
    
# Compare Colleges Route
@app.route("/compare", endpoint="compare")
@app.route("/compare", endpoint="compare_colleges")
@login_required
def compare():
    """
    Compare selected colleges, including the selected branch.

    Supported formats:
        ?college=kcet:1
        ?college=kcet:1:CSE
        ?college=kcet:1:CSE&college=kcet:5:AIML

    Maximum: 3 colleges per comparison.
    """

    # Get selected college values from the URL.
    selected_values = request.args.getlist("college")

    # Backward compatibility with the older indices parameter.
    if not selected_values:
        selected_values = request.args.getlist("indices")

    # Get the default exam.
    default_exam = request.args.get("exam", "KCET").strip().upper()

    if default_exam not in ALLOWED_EXAMS:
        default_exam = "KCET"

    requested_branch = request.args.get("branch", "").strip().upper()
    if requested_branch and requested_branch not in ALLOWED_BRANCHES:
        requested_branch = ""

    colleges = []
    seen = set()

    for value in selected_values:
        value = str(value or "").strip()

        if not value:
            continue

        exam = default_exam
        index_text = value
        selected_branch = requested_branch

        # Parse exam:index or exam:index:branch.
        parts = value.split(":", 2)

        if len(parts) >= 2:
            possible_exam = parts[0].strip().upper()

            if possible_exam in ALLOWED_EXAMS:
                exam = possible_exam
                index_text = parts[1].strip()

                if len(parts) == 3 and not requested_branch:
                    selected_branch = parts[2].strip()

        # Validate the source index.
        try:
            college_index = int(index_text)
        except (TypeError, ValueError):
            continue

        # Prevent duplicate selections of the same college.
        key = (
            exam,
            college_index
        )

        if key in seen:
            continue

        seen.add(key)

        # Retrieve the selected college and branch.
        college = get_college_from_source_index(
            exam,
            college_index,
            selected_branch=selected_branch or None
        )

        # Add only valid colleges.
        if college is not None:
            colleges.append(college)

        # Allow a maximum of three colleges.
        if len(colleges) >= 3:
            break

    # Collect all available branches across compared colleges
    available_branches = []
    for col in colleges:
        for b in col.get("Branches", []):
            b_name = b.get("Branch")
            if b_name and b_name not in available_branches:
                available_branches.append(b_name)

    active_branch = requested_branch if requested_branch in available_branches else (available_branches[0] if available_branches else "CSE")

    # Sync active branch values to college top-level keys
    for col in colleges:
        b_map = col.get("BranchMap", {})
        if active_branch in b_map:
            col["Branch"] = active_branch
            col["Cutoff Rank"] = b_map[active_branch]["Cutoff Rank"]
            col["Exam Fees"] = b_map[active_branch]["Exam Fees"]
            col["Management Fees"] = b_map[active_branch]["Management Fees"]

    # Display the comparison page.
    return render_template(
        "compare.html",
        colleges=colleges,
        saved_colleges=colleges,
        selected_exam=default_exam,
        available_branches=available_branches,
        active_branch=active_branch
    )


# ============================================================
# OLD /templates/ URL COMPATIBILITY
# ============================================================

@app.route("/templates/login")
@app.route("/templates/login.html")
def old_login_url(): return redirect(url_for("dashboard" if session.get("user_id") else "login"))

@app.route("/templates/dashboard")
@app.route("/templates/dashboard.html")
def old_dashboard_url(): return redirect(url_for("dashboard" if session.get("user_id") else "login"))

@app.route("/templates/colleges")
@app.route("/templates/colleges.html")
def old_colleges_url(): return redirect(url_for("colleges" if session.get("user_id") else "login"))

@app.route("/templates/predictor")
@app.route("/templates/predictor.html")
def old_predictor_url(): return redirect(url_for("predictor" if session.get("user_id") else "login"))

@app.route("/templates/saved_colleges")
@app.route("/templates/saved_colleges.html")
def old_saved_url(): return redirect(url_for("saved_colleges" if session.get("user_id") else "login"))

@app.route("/templates/compare")
@app.route("/templates/compare.html")
def old_compare_url(): return redirect(url_for("compare_colleges" if session.get("user_id") else "login"))


# ============================================================
# AI COLLEGE ASSISTANT (GEMINI + RAG CONTEXT)
# ============================================================

COLLEGE_ACRONYMS = {
    "rv": "rv college of engineering",
    "rvce": "rv college of engineering",
    "bms": "bms college of engineering",
    "bmsce": "bms college of engineering",
    "pes": "pes university",
    "pesu": "pes university",
    "pesit": "pes institute of technology",
    "msrit": "ramaiah institute of technology",
    "msr": "ramaiah institute of technology",
    "ramaiah": "ramaiah institute of technology",
    "dsce": "dayananda sagar college of engineering",
    "dayananda": "dayananda sagar college of engineering",
    "nhce": "new horizon college of engineering",
    "new horizon": "new horizon college of engineering",
    "bit": "bangalore institute of technology",
    "uvce": "university visvesvaraya college of engineering",
    "mbu": "mohan babu university",
    "mohan babu": "mohan babu university",
    "svce": "sv college of engineering",
    "svu": "sri venkateswara university",
    "svuce": "sri venkateswara university",
    "cbit": "chaitanya bharathi institute of technology",
    "aits": "annamacharya institute of technology",
    "ksrm": "ksrm college of engineering",
    "gvp": "gayatri vidya parishad college of engineering",
    "gvpce": "gayatri vidya parishad college of engineering",
    "mits": "madanapalli institute of technology",
    "svecw": "shri vishnu engineering college",
    "vishnu": "vishnu institute of technology",
    "jntu": "jntua",
    "jntua": "jntua ananthapur",
    "jntuk": "jntu kakinada",
    "rgm": "rajeev gandhi memorial college",
    "rgmcet": "rajeev gandhi memorial college",
    "gprec": "g pulla reddy engineering college",
    "rvrjc": "rvr & jc college of engineering",
    "srkr": "srkr engineering college",
    "vrsec": "velagapudi ramakrishna siddhartha",
    "siddhartha": "velagapudi ramakrishna siddhartha",
    "vignan": "vignan's institute of information",
    "klu": "kl university",
    "gitam": "gitam university",
    "reva": "reva university",
    "jss": "jss science and technology university",
    "sjce": "jss science and technology university",
    "nie": "the national institute of engineering",
    "kle": "kle technological university",
    "oxford": "the oxford college of engineering",
}

def format_college_row(exam, row):
    c_name = re.sub(r"\s+", " ", str(row.get("College Name", "Unknown"))).strip()
    branch = str(row.get("Branch", "General")).strip()
    cutoff = row.get("Cutoff Rank", "N/A")
    rating = row.get("Rating", "N/A")
    loc = str(row.get("Location", "Karnataka/AP")).strip()
    state = str(row.get("State", "")).strip()
    govt_fees = row.get("Kcet Fees") or row.get("Eamcet Fees") or row.get("Exam Fees") or "N/A"
    mgmt_fees = row.get("Management Fees") or "N/A"
    hostel_fees = row.get("Hostel Fees") or "N/A"
    website = str(row.get("Website", "")).strip()

    def fmt_cur(val):
        try:
            num = float(val)
            return f"₹{num:,.0f}/yr" if num > 0 else "N/A"
        except (ValueError, TypeError):
            return str(val) if val else "N/A"

    g_fmt = fmt_cur(govt_fees)
    m_fmt = fmt_cur(mgmt_fees)
    h_fmt = fmt_cur(hostel_fees)
    loc_str = f"{loc}, {state}" if state and state.lower() not in loc.lower() else loc

    parts = [
        f"• [{exam}] {c_name} ({branch})",
        f"Cutoff: ~{cutoff}",
        f"Rating: {rating}/5",
        f"Location: {loc_str}",
        f"Govt Fees: {g_fmt}",
        f"Mgmt Fees: {m_fmt}",
        f"Hostel Fees: {h_fmt}"
    ]
    if website and website not in ("", "nan", "N/A"):
        parts.append(f"Website: {website}")
    return " | ".join(parts)


GENERIC_COLLEGE_TERMS = {
    "college", "colleges", "engineering", "institute", "institue", "institutes",
    "university", "universities", "technology", "technologies", "school", "academy",
    "group", "institutions", "autonomous"
}

COLLEGE_SEARCH_STOPWORDS = {
    "the", "for", "what", "tell", "about", "and", "cut", "cutoff", "off", "rank", "ranks",
    "how", "much", "is", "are", "in", "at", "to", "can", "get", "with", "my", "please",
    "give", "show", "list", "good", "best", "top", "which", "there", "any", "fees", "fee",
    "hostel", "management", "mgmt", "website", "compare", "versus", "vs", "better", "than",
    "or", "between", "latest", "entered", "search"
}

def retrieve_college_context(query, kcet_df_source, eamcet_df_source, max_results=6):
    """
    Intelligent college context analyzer supporting acronyms, multi-college comparisons,
    locations, branch filtering, cutoffs, and 13 standardized attributes.
    """
    if not query:
        return ""

    q_clean = re.sub(r"\s+", " ", str(query)).strip().lower()
    q_norm = q_clean.replace("bangalore", "bengaluru").replace("vizag", "visakhapatnam").replace("tirupathi", "tirupati")

    # Detect branch filter if specified
    branch_map = {"cse": "CSE", "aiml": "AIML", "ece": "ECE", "eee": "EEE", "me": "ME", "mechanical": "ME", "civil": "Civil", "ise": "ISE"}
    found_branch = None
    for b_key, b_val in branch_map.items():
        if re.search(r"\b" + re.escape(b_key) + r"\b", q_norm):
            found_branch = b_val
            break

    # Detect rank range
    rank_matches = re.findall(r"\b\d{3,6}\b", q_norm)
    target_rank = int(rank_matches[0]) if rank_matches else None

    is_asking_best = any(k in q_norm for k in ["best", "top", "highest", "ranking", "reputed", "famous", "good"])
    is_comparison = any(k in q_norm for k in ["compare", " vs ", " versus ", " or ", "better than"])

    def run_subsearch(sub_q, limit):
        tokens = re.findall(r"[a-z0-9]+", sub_q)
        acronym_targets = [COLLEGE_ACRONYMS[t] for t in tokens if t in COLLEGE_ACRONYMS]
        search_terms = [t for t in tokens if t not in COLLEGE_SEARCH_STOPWORDS and t not in GENERIC_COLLEGE_TERMS and (len(t) >= 3 or t in ("rv", "ms", "sv", "au", "me"))]

        sub_res = []
        seen = set()

        def add_matching_rows(df, exam, mask):
            if df is None or df.empty:
                return
            matched = df[mask].copy()
            if matched.empty:
                return
            if found_branch:
                b_match = matched[matched["Branch"].str.upper() == found_branch.upper()]
                if not b_match.empty:
                    matched = b_match
            if is_asking_best:
                matched = matched.sort_values(by=["Rating", "Cutoff Rank"], ascending=[False, True])
            else:
                matched = matched.sort_values(by="Cutoff Rank", ascending=True)

            for _, row in matched.iterrows():
                key = (exam, row["College Name"], row.get("Branch", ""))
                if key not in seen:
                    seen.add(key)
                    sub_res.append((exam, row))
                    if len(sub_res) >= limit:
                        return

        def get_norm_series(df):
            name_s = df["College Name"].astype(str).str.replace(r"\s+", " ", regex=True).str.lower()
            loc_s = df["Location"].astype(str).str.replace(r"\s+", " ", regex=True).str.lower()
            return name_s, loc_s

        # 1. Acronym matches
        for acr in acronym_targets:
            for exam, df in [("EAMCET", eamcet_df_source), ("KCET", kcet_df_source)]:
                if df is None or df.empty:
                    continue
                n_s, _ = get_norm_series(df)
                add_matching_rows(df, exam, n_s.str.contains(acr, na=False))
                if len(sub_res) >= limit:
                    return sub_res

        # 2. Multi-word phrase matches (e.g. 'mohan babu')
        if len(search_terms) >= 2:
            phrase = " ".join(search_terms)
            for exam, df in [("EAMCET", eamcet_df_source), ("KCET", kcet_df_source)]:
                if df is None or df.empty:
                    continue
                n_s, _ = get_norm_series(df)
                add_matching_rows(df, exam, n_s.str.contains(phrase, na=False))
                if len(sub_res) >= limit:
                    return sub_res

        # 3. Location matches
        for term in search_terms:
            pattern = r"\b" + re.escape(term) + r"\b" if len(term) <= 3 else term
            for exam, df in [("EAMCET", eamcet_df_source), ("KCET", kcet_df_source)]:
                if df is None or df.empty:
                    continue
                _, l_s = get_norm_series(df)
                add_matching_rows(df, exam, l_s.str.contains(pattern, regex=True, na=False))
                if len(sub_res) >= limit:
                    return sub_res

        # 4. College Name term match (using word boundary for short tokens)
        for term in search_terms:
            pattern = r"\b" + re.escape(term) + r"\b" if len(term) <= 3 else term
            for exam, df in [("EAMCET", eamcet_df_source), ("KCET", kcet_df_source)]:
                if df is None or df.empty:
                    continue
                n_s, _ = get_norm_series(df)
                add_matching_rows(df, exam, n_s.str.contains(pattern, regex=True, na=False))
                if len(sub_res) >= limit:
                    return sub_res

        # 5. Rank range match if specified
        if target_rank and len(sub_res) < limit:
            min_r = int(target_rank * 0.70)
            max_r = int(target_rank * 1.35)
            for exam, df in [("KCET", kcet_df_source), ("EAMCET", eamcet_df_source)]:
                if df is None or df.empty:
                    continue
                add_matching_rows(df, exam, (df["Cutoff Rank"] >= min_r) & (df["Cutoff Rank"] <= max_r))
                if len(sub_res) >= limit:
                    return sub_res

        return sub_res

    results = []
    if is_comparison:
        parts = re.split(r"\b(?:vs|versus|and|or|better than|compared to)\b", q_norm, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            r1 = run_subsearch(parts[0], limit=2)
            r2 = run_subsearch(parts[1], limit=2)
            results = r1[:2] + r2[:2]

    if len(results) < max_results:
        rem = run_subsearch(q_norm, limit=max_results - len(results))
        results.extend(rem)

    return "\n".join([format_college_row(exam, row) for exam, row in results[:max_results]])


def get_user_prediction_context(user_id):
    """
    Fetch logged-in student's recent prediction queries and entered ranks from database for both KCET & EAMCET.
    """
    if not user_id:
        return ""
    try:
        conn = get_db_connection()
        eamcet_rows = conn.execute(
            "SELECT exam, rank, branch, category, created_at FROM prediction_history WHERE user_id = %s AND LOWER(exam) = 'eamcet' ORDER BY id DESC LIMIT 3",
            (user_id,)
        ).fetchall()
        kcet_rows = conn.execute(
            "SELECT exam, rank, branch, category, created_at FROM prediction_history WHERE user_id = %s AND LOWER(exam) = 'kcet' ORDER BY id DESC LIMIT 3",
            (user_id,)
        ).fetchall()
        conn.close()

        lines = ["Logged-in Student's Search History:"]
        if eamcet_rows:
            lines.append("Recent EAMCET Searches:")
            for r in eamcet_rows:
                lines.append(f"• EAMCET: Rank {r.get('rank')}, Branch {r.get('branch', 'General')} (Date: {str(r.get('created_at', ''))[:10]})")
        if kcet_rows:
            lines.append("Recent KCET Searches:")
            for r in kcet_rows:
                lines.append(f"• KCET: Rank {r.get('rank')}, Branch {r.get('branch', 'General')} (Date: {str(r.get('created_at', ''))[:10]})")

        return "\n".join(lines)
    except Exception as exc:
        app.logger.warning("Error fetching user prediction context: %s", exc)
        return ""


def generate_fallback_reply(message, context, user_context=""):
    """
    Context-aware short intelligent response (2-4 lines) analyzing college attributes,
    fees, cutoffs, websites, comparisons, and project data when external Gemini call is offline.
    """
    msg_lower = message.lower()

    # 1. User asking about their entered rank or history
    if any(k in msg_lower for k in ["my rank", "latest rank", "i entered", "entered in", "my search", "my history", "previous search"]):
        if user_context and "•" in user_context:
            first_line = user_context.split("•")[1].split("\n")[0].strip()
            return f"Your latest entered search: **{first_line}**.\nYou can review your full history on your [Prediction History](/prediction_history) page."
        return "I couldn't find any recent search entries in your account yet. Try using the [College Predictor](/predictor) to run your first prediction!"

    # 2. Machine Learning Model & Accuracy questions
    if any(k in msg_lower for k in ["ml model", "machine learning", "accuracy", "random forest", "algorithm", "model performance"]):
        return (
            "🤖 **Machine Learning Architecture:**\n"
            "This project uses a Scikit-learn **Random Forest Classifier** (`model/college_predictor.pkl`) trained on historical round allotment distributions.\n"
            "• **Accuracy:** **90.81%** | **Precision:** **93.02%** | **Recall:** **91.56%**"
        )

    # 3. Datasets & Total Colleges questions
    if any(k in msg_lower for k in ["dataset", "datasets", "how many colleges", "records", "data in project", "attribute"]):
        return (
            "📊 **Project Datasets Overview:**\n"
            "• **Total Programs:** **1,331 accredited offerings** across 13 standardized attributes.\n"
            "• **KCET Dataset:** 560 engineering institutions in Karnataka (`dataset/kcet_colleges.csv`).\n"
            "• **EAMCET Dataset:** 771 engineering institutions in AP/Telangana (`dataset/eamcet_colleges.csv`)."
        )

    # 4. Tech Stack & Technologies used
    if any(k in msg_lower for k in ["tech stack", "technology", "technologies", "framework", "built with", "backend", "frontend"]):
        return (
            "⚙️ **System Tech Stack:**\n"
            "• **Backend & ML:** Python 3, Flask framework, Scikit-learn Random Forest, Pandas.\n"
            "• **Database & Cloud:** PostgreSQL on Render Cloud + SQLite local fallback (`college_predictor.db`).\n"
            "• **Frontend & AI:** Responsive Tailwind CSS SaaS UI + Google Gemini 3.5 AI Assistant."
        )

    # 5. Automation & Real-time counseling updates
    if any(k in msg_lower for k in ["automation", "yearly", "batch", "crawler", "update dataset"]):
        return (
            "🔄 **Yearly Automation Engine:**\n"
            "The platform includes `run_yearly_automation.bat` and python crawler scripts in `automation/` to synchronize real-time KEA/APSCHE cutoffs, scrape NAAC ratings, and retrain ML models annually."
        )

    # 6. Admin, Developer & Team inquiries
    if any(k in msg_lower for k in [
        "admin", "administrator", "system admin", "support phone", "admin phone",
        "developer", "developer phone", "contact admin", "contact number",
        "phone number", "linkedin", "linkdin", "github", "uday", "creator", "founder", "team", "coordinator"
    ]):
        return (
            "👨‍💻 **System Administrator & Lead Developer: Uday**\n"
            "• **Phone:** [+91 8919221139](tel:8919221139) | **Email:** [udayu6853@gmail.com](mailto:udayu6853@gmail.com)\n"
            "• **LinkedIn:** [Uday on LinkedIn](https://www.linkedin.com/in/uday-uday-270441354) | **GitHub:** [udayu6853-jpg](https://github.com/udayu6853-jpg)\n"
            "• **Project Coordinators:** Varun, Prajwal, Pruthvi | **Portal:** [/admin-contact](/admin-contact)"
        )

    # 7. Counseling inquiry
    if any(k in msg_lower for k in ["counseling", "counselling", "document", "documents", "procedure", "verification", "process"]):
        return (
            "📋 **Key Counseling Documents Checklist:**\n"
            "1. Rank Card & Admit Card (KCET / EAMCET)\n"
            "2. 10th & 12th Marks Cards + Study Certificate\n"
            "3. Category / Domicile Certificate (if claiming reservation)\n"
            "Follow official portal instructions for online option entry."
        )

    # 8. Detailed College Information Analysis from Context (ZERO UNWANTED TEXT)
    if context.strip():
        lines = [line.strip() for line in context.strip().split("\n") if line.strip().startswith("•")]
        parsed_colleges = []
        for l in lines:
            p = {}
            parts = [x.strip() for x in l.split("|")]
            p["title"] = parts[0].replace("•", "").strip()
            for pt in parts[1:]:
                if ":" in pt:
                    k, v = pt.split(":", 1)
                    p[k.strip().lower()] = v.strip()
            parsed_colleges.append(p)

        if parsed_colleges:
            top = parsed_colleges[0]

            # A. Hostel fees analysis (ONLY hostel fee)
            if "hostel" in msg_lower:
                h_fee = top.get("hostel fees", "N/A")
                loc = top.get("location", "the campus region")
                return f"🏨 **Hostel Fee:** **{top['title']}** is **{h_fee}** ({loc})."

            # B. Management quota fees analysis (ONLY management & govt fees)
            if any(k in msg_lower for k in ["management fee", "management fees", "mgmt fee", "mgmt fees"]):
                m_fee = top.get("mgmt fees", "N/A")
                g_fee = top.get("govt fees", "N/A")
                return f"💰 **Management Quota Fee:** **{top['title']}** is **{m_fee}** (Govt Convener Fee: {g_fee})."

            # C. Official Website analysis (ONLY website link)
            if any(k in msg_lower for k in ["website", "link", "portal", "url", "site"]):
                web = top.get("website", "")
                if web and web != "N/A":
                    return f"🌐 **Official Website:** **{top['title']}**: [{web}]({web})"

            # D. Side-by-side College Comparison (ONLY comparative facts)
            if any(k in msg_lower for k in ["compare", " vs ", " versus ", " or ", "better than"]) and len(parsed_colleges) >= 2:
                c1 = parsed_colleges[0]
                c1_clean = c1["title"].split("(")[0].strip()
                c2 = next((c for c in parsed_colleges[1:] if c["title"].split("(")[0].strip() != c1_clean), parsed_colleges[1])
                return (
                    f"⚖️ **College Comparison:**\n"
                    f"• **{c1['title']}:** Rating {c1.get('rating', 'N/A')} | Cutoff {c1.get('cutoff', 'N/A')} | Govt {c1.get('govt fees', 'N/A')} | Hostel {c1.get('hostel fees', 'N/A')}\n"
                    f"• **{c2['title']}:** Rating {c2.get('rating', 'N/A')} | Cutoff {c2.get('cutoff', 'N/A')} | Govt {c2.get('govt fees', 'N/A')} | Hostel {c2.get('hostel fees', 'N/A')}"
                )

            # E. Best / Top Institution in Location (ONLY top college facts)
            if any(k in msg_lower for k in ["best", "top", "highest rating", "reputed"]):
                return (
                    f"🏆 **Top College in {top.get('location', 'the region')}:** **{top['title']}**\n"
                    f"• Rating: **{top.get('rating', 'N/A')}** | Cutoff: **{top.get('cutoff', 'N/A')}** | Govt Fees: **{top.get('govt fees', 'N/A')}** | Mgmt Fees: **{top.get('mgmt fees', 'N/A')}**"
                )

            # F. Specific branch cutoff inquiry (ONLY cutoff rank)
            if any(k in msg_lower for k in ["cutoff", "cut off", "closing rank"]):
                return f"🎯 **Cutoff Rank:** **{top['title']}** closes at **{top.get('cutoff', 'N/A')}** (Rating: {top.get('rating', 'N/A')})."

            # G. General matched college list (ONLY matched colleges)
            top_colleges = "\n".join(lines[:3])
            return f"🎓 **Matching Colleges:**\n{top_colleges}"

    return (
        "👋 Hi! I can help you with KCET/EAMCET cutoffs, ratings, govt & management fees, hostel fees, or your search history.\n"
        "What college, city, or branch would you like to explore?"
    )


def call_gemini_api(api_key, message, history, context):
    """
    Call Google Gemini REST API with strict rules against unwanted text, conversational filler, or unrequested info.
    """
    models = [
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-1.5-flash",
        "gemini-pro-latest"
    ]
    system_instruction = (
        "You are the official Smart College Predictor AI Assistant for KCET and EAMCET.\n"
        "CRITICAL INSTRUCTION - ABSOLUTELY NO UNWANTED TEXT:\n"
        "1. ANSWER ONLY WHAT WAS SPECIFICALLY ASKED: Output strictly the exact factual data requested. Never add unsolicited details, unwanted website links, extra fees, or unrequested information.\n"
        "   - If asked for hostel fee -> Output ONLY the annual hostel fee (e.g. 'The annual hostel fee for [College] is ₹X/yr.'). Do NOT add website links, tuition fees, or campus descriptions.\n"
        "   - If asked for website -> Output ONLY the official website URL. Do NOT add advice or syllabus info.\n"
        "   - If asked for management fee -> Output ONLY the management quota fee (and govt fee). Do NOT add hostel or website links.\n"
        "   - If asked for cutoff -> Output ONLY the cutoff rank and branch.\n"
        "   - If asked for phone number -> Output ONLY the phone number.\n"
        "   - If asked for LinkedIn -> Output ONLY the LinkedIn URL.\n"
        "   - If asked for best college in a city -> Name the top institution, rating, cutoff, and fees in 1-2 lines.\n"
        "2. ZERO CONVERSATIONAL FILLER & ZERO PLEASANTRIES: Never include greetings, pleasantries, or filler phrases (e.g. 'Certainly!', 'Sure!', 'Here is the information:', 'Hope this helps!', 'For more details visit...', 'Feel free to ask!'). Start immediately with the direct answer.\n"
        "3. NO MARKETING / NO PROMPT SUGGESTIONS: Never tell the user to bookmark, visit dashboard, or check links unless explicitly asked.\n"
        "4. STRICTLY 1 TO 2 LINES MAXIMUM: Provide clean, direct, factual answers.\n\n"
        "KNOWLEDGE BASE:\n"
        "• Project: Smart College Predictor (college-predictor-ml.onrender.com) for KCET & EAMCET.\n"
        "• Datasets: 1,331 accredited college programs across 13 attributes (560 KCET, 771 EAMCET).\n"
        "• ML Model: RandomForestClassifier (90.81% accuracy, 93.02% precision, 91.56% recall, model/college_predictor.pkl).\n"
        "• Tech Stack: Python 3, Flask, Scikit-learn, PostgreSQL on Render + SQLite local fallback, Tailwind CSS.\n"
        "• Administrator: Uday (Phone: +91 8919221139, Email: udayu6853@gmail.com, LinkedIn: https://www.linkedin.com/in/uday-uday-270441354, GitHub: https://github.com/udayu6853-jpg).\n"
        "• User Search History: ONLY state past search history if the student explicitly asks 'what is my rank', 'what was my search', or 'my history'. If the user did not explicitly ask for their search history, NEVER mention past rank, exam, or search records."
    )

    full_prompt = f"Student Question: {message}\n"
    if context:
        full_prompt = f"Relevant Official Context from Database:\n{context}\n\n" + full_prompt

    contents = []
    # Avoid duplicate current message if widget already appended it to history
    valid_history = [
        item for item in (history or [])
        if not (item.get("role") == "user" and str(item.get("content", "")).strip() == message)
    ]

    for item in valid_history[-4:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})

    contents.append({"role": "user", "parts": [{"text": full_prompt}]})

    for model_name in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": contents,
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                },
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 95
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
        except urllib.error.HTTPError as he:
            app.logger.warning("Gemini model %s returned HTTP %s: %s", model_name, he.code, he.reason)
            if he.code == 404:
                continue
            break
        except Exception as exc:
            app.logger.warning("Gemini model %s request exception: %s", model_name, exc)
            break

    return None


@app.route("/api/chat", methods=["POST"])
@csrf.exempt
@limiter.limit("20 per minute")
def api_chat():
    """
    JSON API endpoint for the AI College Assistant chatbot widget.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "error": "Authentication required. Please sign in to consult the AI Assistant."}), 401

    load_dotenv(override=True)
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()[:1000]
    history = data.get("history", [])
    if isinstance(history, list):
        history = history[-8:]
    else:
        history = []

    if not message:
        return jsonify({"status": "error", "error": "Message cannot be empty."}), 400

    # 1. Reject gibberish / nonsense (e.g. "gg", "asdf", "...", random repeated keys)
    clean_msg = re.sub(r"[^a-zA-Z0-9\s]", "", message).strip().lower()
    valid_short_abbrs = {"cs", "ce", "ee", "me", "ec", "ai", "is", "it", "bt"}
    is_gibberish = False
    if len(clean_msg) < 3 and clean_msg not in valid_short_abbrs:
        is_gibberish = True
    elif len(set(clean_msg.replace(" ", ""))) <= 1:
        is_gibberish = True
    elif re.match(r"^[bcdfghjklmnpqrstvwxyz]{3,}$", clean_msg):
        is_gibberish = True

    if is_gibberish:
        return jsonify({
            "status": "success",
            "reply": "Please ask a specific question about colleges, cutoffs, fees, or rankings (e.g. *'Best college in Tirupati'* or *'Top KCET colleges for CSE'*)."
        })

    user_context = get_user_prediction_context(user_id)
    college_context = retrieve_college_context(message, kcet_df, eamcet_df)

    context_parts = []
    # ONLY inject user prediction history if student EXPLICITLY asks about their rank or previous search!
    is_asking_user_history = any(
        k in message.lower()
        for k in [
            "my rank", "my search", "my history", "entered rank", "latest rank",
            "previous search", "past search", "what did i search", "what was my rank",
            "i entered", "searched before"
        ]
    )
    if is_asking_user_history and user_context:
        context_parts.append(user_context)

    if college_context:
        context_parts.append(college_context)
    combined_context = "\n\n".join(context_parts)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    reply = None
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            reply = call_gemini_api(api_key, message, history, combined_context)
        except Exception as exc:
            app.logger.warning("Gemini API call error: %s", exc)

    if not reply:
        reply = generate_fallback_reply(message, combined_context, user_context)

    return jsonify({"status": "success", "reply": reply})


# ============================================================
# ACTIVE INTRUSION & SCANNER SHIELD (WAF)
# ============================================================

BLOCKED_SCANNER_PATTERNS = (
    ".env", ".git", ".aws", ".config", "wp-admin", "wp-login", "xmlrpc.php",
    "phpmyadmin", "eval-stdin.php", "phpunit", "boaform", "cgi-bin",
    ".sql", ".bak", "/etc/passwd", "win.ini"
)

@app.before_request
def block_malicious_probes():
    path_lower = request.path.lower()

    # 1. Block common hacker reconnaissance scanners
    for pattern in BLOCKED_SCANNER_PATTERNS:
        if pattern in path_lower and not path_lower.endswith((".js", ".css", ".png", ".jpg", ".webp")):
            app.logger.warning(
                "Security Shield: Blocked malicious scan from IP=%s for path=%s",
                request.remote_addr, request.path
            )
            return render_template("403.html"), 403

    # 2. Block path traversal attempts
    if ".." in request.path or "%2e%2e" in path_lower:
        app.logger.warning(
            "Security Shield: Blocked path traversal attempt from IP=%s: %s",
            request.remote_addr, request.path
        )
        return render_template("403.html"), 403


# ============================================================
# SECURITY HEADERS / ERRORS
# ============================================================

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(self), camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' 'unsafe-inline' 'unsafe-eval' https: http: data: blob:; "
        "frame-ancestors 'self';"
    )
    if "text/html" in response.headers.get("Content-Type", ""):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@app.errorhandler(403)
def forbidden(error):
    return render_template("403.html"), 403

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return ("Request is too large. Please upload an image smaller than 5 MB.", 413)

@app.errorhandler(429)
def ratelimit_handler(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "reply": "Rate limit reached. For security and anti-spam protection, please wait a minute before sending another query."
        }), 429
    return render_template(
        "429.html",
        message="Too many requests detected. For your security and server protection against brute-force attacks, please wait a minute before trying again."
    ), 429

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Unhandled server error: %s", error)
    return render_template("500.html"), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    is_debug = os.getenv("FLASK_DEBUG", "false").strip().lower() in ("true", "1", "yes")
    app.run(host="127.0.0.1", port=5000, debug=is_debug)