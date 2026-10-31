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
    flash
)

import pandas as pd
import os
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

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing."
    )


def get_db_connection():

    connection = psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row
    )

    return connection

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                SELECT id
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
# LOGIN PAGE
# ============================================================

@app.route("/login")
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    return render_template("login.html")


# ============================================================
# GOOGLE LOGIN
# ============================================================

@app.route("/google/login")
def google_login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    try:
        return google.authorize_redirect(GOOGLE_REDIRECT_URI)
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
def signup():
    return redirect(url_for("login"))


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
# SECURITY HEADERS / ERRORS
# ============================================================

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["X-Frame-Options"]="SAMEORIGIN"
    response.headers["Referrer-Policy"]="strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]="geolocation=(), microphone=(), camera=()"
    return response

@app.errorhandler(404)
def page_not_found(error):
    return redirect(url_for("dashboard" if session.get("user_id") else "login"))

@app.errorhandler(413)
def request_entity_too_large(error):
    return ("Request is too large. Please upload an image smaller than 5 MB.", 413)

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Unhandled server error")
    return "An unexpected server error occurred.",500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug=True)