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
    1 * 1024 * 1024
)


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
            r"[â‚¹$]",
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


# ============================================================
# DATABASE CONNECTION
# ============================================================

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

@app.route(
    "/dashboard"
)
@login_required
def dashboard():

    user_id = session.get(
        "user_id"
    )

    connection = get_db_connection()

    user = connection.execute(

        """
        SELECT *
        FROM users
        WHERE id = %s
        """,

        (
            user_id,
        )

    ).fetchone()

    history = connection.execute(

        """
        SELECT *
        FROM prediction_history
        WHERE user_id = %s
        ORDER BY created_at DESC
        """,

        (
            user_id,
        )

    ).fetchall()

    connection.close()

    if user is None:

        session.clear()

        return redirect(
            url_for(
                "login"
            )
        )

    total_predictions = len(
        history
    )

    kcet_predictions = sum(
        1
        for item in history
        if item["exam"] == "KCET"
    )

    eamcet_predictions = sum(
        1
        for item in history
        if item["exam"] == "EAMCET"
    )

    recent_predictions = history[
        :5
    ]

    return render_template(

        "dashboard.html",

        user=user,

        total_colleges=(
            len(kcet_df)
            +
            len(eamcet_df)
        ),

        kcet_colleges=len(
            kcet_df
        ),

        eamcet_colleges=len(
            eamcet_df
        ),

        total_predictions=(
            total_predictions
        ),

        kcet_predictions=(
            kcet_predictions
        ),

        eamcet_predictions=(
            eamcet_predictions
        ),

        recent_predictions=(
            recent_predictions
        )
    )


# ============================================================
# COLLEGES
# ============================================================

@app.route(
    "/colleges"
)
@login_required
def colleges():

    exam = request.args.get(
        "exam",
        "all"
    ).strip().lower()

    if exam == "kcet":

        dataframe = kcet_df.copy()

        selected_exam = "KCET"

        offset = 0

    elif exam == "eamcet":

        dataframe = eamcet_df.copy()

        selected_exam = "EAMCET"

        offset = len(
            kcet_df
        )

    else:

        dataframe = pd.concat(
            [
                kcet_df,
                eamcet_df
            ],
            ignore_index=True
        )

        selected_exam = "ALL"

        offset = 0

    college_list = []

    for index, row in dataframe.iterrows():

        college = row.to_dict()

        if selected_exam == "ALL":

            if str(
                row.get(
                    "Exam",
                    ""
                )
            ).upper() == "EAMCET":

                source_index = (
                    index
                    -
                    len(kcet_df)
                )

            else:

                source_index = index

        else:

            source_index = index

        college["SourceIndex"] = (
            source_index
        )

        college["Exam"] = str(
            row.get(
                "Exam",
                selected_exam
            )
        ).upper()

        college["Website"] = clean_website(

            row.get(
                "Website",
                ""
            ),

            row.get(
                "College Name",
                ""
            )
        )

        college["Image"] = clean_image(
            row.get(
                "Image",
                ""
            )
        )

        # Correct exam-specific fee
        if college["Exam"] == "KCET":

            college["Exam Fees"] = (
                clean_numeric(
                    row.get(
                        "Kcet Fees",
                        row.get(
                            "KCET Fees",
                            0
                        )
                    )
                )
            )

        else:

            college["Exam Fees"] = (
                clean_numeric(
                    row.get(
                        "Eamcet Fees",
                        row.get(
                            "EAMCET Fees",
                            0
                        )
                    )
                )
            )

        college_list.append(
            college
        )

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
# COMPLETE PREDICTION RESULTS
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
@limiter.limit("20 per minute")
def predict():

    # ========================================================
    # GET STUDENT INPUT
    # ========================================================

    exam = (
        request.form.get("exam", "")
        .strip()
        .upper()
    )

    rank_text = (
        request.form.get("rank", "")
        .strip()
    )

    branch = (
        request.form.get("branch", "")
        .strip()
        .upper()
    )

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    valid, validation_error = (
        validate_prediction_input(
            exam,
            rank_text,
            branch
        )
    )

    if not valid:

        return render_template(
            "predictor.html",
            branches=sorted(
                ALLOWED_BRANCHES
            ),
            error=validation_error,
            selected_exam=exam,
            selected_rank=rank_text,
            selected_branch=branch
        )

    # ========================================================
    # CONVERT RANK
    # ========================================================

    rank = int(rank_text)

    # ========================================================
    # SELECT DATASET
    # ========================================================

    if exam == "KCET":

        dataframe = kcet_df.copy()

    else:

        dataframe = eamcet_df.copy()

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if dataframe.empty:

        return render_template(
            "result.html",
            colleges=[],
            results=[],
            exam=exam,
            exam_name=exam,
            rank=rank,
            student_rank=rank,
            branch=branch,
            error="No college data is available."
        )

    # ========================================================
    # FILTER BY BRANCH
    # ========================================================

    branch_data = dataframe[
        dataframe["Branch"]
        .astype(str)
        .str.upper()
        .str.strip()
        == branch
    ].copy()

    if branch_data.empty:

        return render_template(
            "result.html",
            colleges=[],
            results=[],
            exam=exam,
            exam_name=exam,
            rank=rank,
            student_rank=rank,
            branch=branch,
            error=(
                f"No colleges found for {branch}."
            )
        )

    # ========================================================
    # SELECT RELEVANT CUTOFF RANGE
    # ========================================================

    min_rank = max(
        1,
        rank - 10000
    )

    max_rank = (
        rank + 10000
    )

    ml_input = branch_data[
        (
            branch_data["Cutoff Rank"]
            >= min_rank
        )
        &
        (
            branch_data["Cutoff Rank"]
            <= max_rank
        )
    ].copy()

    # ========================================================
    # IF NO COLLEGES IN RANGE
    # USE COMPLETE BRANCH DATA
    # ========================================================

    if ml_input.empty:

        ml_input = branch_data.copy()

    # ========================================================
    # ADD STUDENT RANK
    # ========================================================

    ml_input["Student Rank"] = rank

    # ========================================================
    # MAKE SURE REQUIRED NUMERIC COLUMNS EXIST
    # ========================================================

    for column in [
        "Rating",
        "Management Fees",
        "Hostel Fees",
        "Cutoff Rank"
    ]:

        if column not in ml_input.columns:

            ml_input[column] = 0

    # ========================================================
    # CLEAN NUMERIC VALUES
    # ========================================================

    ml_input["Cutoff Rank"] = (
        ml_input["Cutoff Rank"]
        .apply(clean_rank)
    )

    ml_input["Rating"] = (
        ml_input["Rating"]
        .apply(clean_rating)
    )

    ml_input["Management Fees"] = (
        ml_input["Management Fees"]
        .apply(clean_numeric)
    )

    ml_input["Hostel Fees"] = (
        ml_input["Hostel Fees"]
        .apply(clean_numeric)
    )

    # ========================================================
    # EXAM FEES
    # ========================================================

    if "Exam Fees" not in ml_input.columns:

        if exam == "KCET":

            fee_column = "Kcet Fees"

        else:

            fee_column = "Eamcet Fees"

        if fee_column in ml_input.columns:

            ml_input["Exam Fees"] = (
                ml_input[fee_column]
                .apply(clean_numeric)
            )

        else:

            ml_input["Exam Fees"] = 0

    else:

        ml_input["Exam Fees"] = (
            ml_input["Exam Fees"]
            .apply(clean_numeric)
        )

    # ========================================================
    # CALCULATE FEE SCORE
    # ========================================================

    max_exam_fee = float(
        ml_input["Exam Fees"].max()
    )

    if max_exam_fee > 0:

        ml_input["Fee Score"] = (
            1
            - (
                ml_input["Exam Fees"]
                / max_exam_fee
            )
        ).clip(
            0,
            1
        )

    else:

        ml_input["Fee Score"] = 0.5

    # ========================================================
    # CREATE RESULTS
    # ========================================================

    results = []

    # ========================================================
    # RUN PREDICTION FOR EACH COLLEGE
    # ========================================================

    for index, row in ml_input.iterrows():

        # ----------------------------------------------------
        # CLEAN COLLEGE DATA
        # ----------------------------------------------------

        cutoff = clean_rank(
            row.get(
                "Cutoff Rank",
                0
            )
        )

        rating = clean_rating(
            row.get(
                "Rating",
                0
            )
        )

        fee_score = float(
            row.get(
                "Fee Score",
                0.5
            )
        )

        ml_score = None

        # ----------------------------------------------------
        # MACHINE LEARNING PREDICTION
        # ----------------------------------------------------

        if model is not None:

            try:

                # ====================================================
                # IMPORTANT:
                #
                # These columns MUST match train_model.py exactly.
                #
                # TRAINING FEATURES:
                #
                # Exam
                # Branch
                # Student Rank
                # Rating
                # Management Fees
                # Hostel Fees
                # Cutoff Rank
                # ====================================================

                feature_row = pd.DataFrame(
                    [
                        {
                            "Exam": exam,

                            "Branch": branch,

                            "Student Rank": rank,

                            "Rating": rating,

                            "Management Fees": (
                                clean_numeric(
                                    row.get(
                                        "Management Fees",
                                        0
                                    )
                                )
                            ),

                            "Hostel Fees": (
                                clean_numeric(
                                    row.get(
                                        "Hostel Fees",
                                        0
                                    )
                                )
                            ),

                            "Cutoff Rank": cutoff
                        }
                    ]
                )

                # ------------------------------------------------
                # PREDICT PROBABILITY
                # ------------------------------------------------

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            feature_row
                        )
                    )

                    if (
                        len(probabilities) > 0
                        and
                        len(probabilities[0]) >= 2
                    ):

                        # Probability of Eligible = 1
                        ml_score = float(
                            probabilities[0][1]
                        )

                    elif (
                        len(probabilities) > 0
                        and
                        len(probabilities[0]) == 1
                    ):

                        ml_score = float(
                            probabilities[0][0]
                        )

                # ------------------------------------------------
                # FALLBACK TO PREDICT()
                # ------------------------------------------------

                elif hasattr(
                    model,
                    "predict"
                ):

                    prediction = (
                        model.predict(
                            feature_row
                        )[0]
                    )

                    ml_score = float(
                        prediction
                    )

            except Exception as model_error:

                app.logger.warning(
                    "ML prediction fallback used: %s",
                    model_error
                )

                ml_score = None

        # ====================================================
        # RULE-BASED FALLBACK
        # ====================================================

        if ml_score is None:

            if cutoff > 0:

                ml_score = max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - (
                            abs(
                                cutoff - rank
                            )
                            /
                            max(
                                cutoff,
                                rank,
                                1
                            )
                        )
                    )
                )

            else:

                ml_score = 0.5

        # ====================================================
        # KEEP SCORE BETWEEN 0 AND 1
        # ====================================================

        ml_score = max(
            0.0,
            min(
                1.0,
                float(
                    ml_score
                )
            )
        )

        # ====================================================
        # RECOMMENDATION SCORE
        # ====================================================

        recommendation_score = (
            calculate_recommendation_score(
                rank,
                cutoff,
                ml_score,
                rating,
                fee_score
            )
        )

        # ====================================================
        # CONVERT COLLEGE ROW TO DICTIONARY
        # ====================================================

        college = row.to_dict()

        # ====================================================
        # ADD PREDICTION INFORMATION
        # ====================================================

        college.update(
            {
                "Exam": exam,

                "SourceIndex": int(
                    index
                ),

                "Original Index": int(
                    index
                ),

                "ML Probability": round(
                    ml_score * 100,
                    1
                ),

                "Recommendation Score":
                    recommendation_score,

                "Chance": round(
                    recommendation_score * 100,
                    1
                )
            }
        )

        # ====================================================
        # CLEAN WEBSITE
        # ====================================================

        college["Website"] = (
            clean_website(
                college.get(
                    "Website",
                    ""
                ),
                college.get(
                    "College Name",
                    ""
                )
            )
        )

        # ====================================================
        # CLEAN IMAGE
        # ====================================================

        college["Image"] = (
            clean_image(
                college.get(
                    "Image",
                    ""
                )
            )
        )

        # ====================================================
        # CLEAN EXAM FEES
        # ====================================================

        college["Exam Fees"] = (
            clean_numeric(
                college.get(
                    "Exam Fees",
                    0
                )
            )
        )

        # ====================================================
        # ADD RESULT
        # ====================================================

        results.append(
            college
        )

    # ========================================================
    # SORT RESULTS
    # ========================================================

    results.sort(
        key=lambda x: (
            float(
                x.get(
                    "Recommendation Score",
                    0
                )
            ),
            float(
                x.get(
                    "Rating",
                    0
                )
            )
        ),
        reverse=True
    )

    # ========================================================
    # LIMIT RESULTS
    # ========================================================

    results = results[:50]

    # ========================================================
    # SAVE PREDICTION HISTORY
    # ========================================================

    connection = None

    try:

        connection = (
            get_db_connection()
        )

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
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                session.get(
                    "user_id"
                ),

                exam,

                rank,

                branch,

                request.form.get(
                    "category",
                    ""
                ).strip(),

                request.form.get(
                    "gender",
                    ""
                ).strip(),

                request.form.get(
                    "preferred_location",
                    ""
                ).strip(),

                request.form.get(
                    "state",
                    ""
                ).strip(),

                request.form.get(
                    "fee_range",
                    ""
                ).strip()
            )
        )

        connection.commit()

    except Exception as error:

        app.logger.exception(
            "Prediction history save failed: %s",
            error
        )

    finally:

        if connection is not None:

            connection.close()

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return render_template(
        "result.html",

        colleges=results,

        results=results,

        exam=exam,

        exam_name=exam,

        rank=rank,

        student_rank=rank,

        branch=branch
    )

# ============================================================
# PROFILE / HELP / ABOUT / HISTORY
# ============================================================

@app.route("/profile")
@login_required
def profile():
    connection=get_db_connection(); user=connection.execute("SELECT id,name,email,created_at FROM users WHERE id=%s",(session.get("user_id"),)).fetchone(); connection.close()
    if user is None: session.clear(); return redirect(url_for("login"))
    return render_template("profile.html",user=user)

@app.route("/update-profile",methods=["POST"])
@login_required
def update_profile():
    name=request.form.get("name","").strip()[:100]
    if not name: flash("Name cannot be empty.","error"); return redirect(url_for("profile"))
    connection=get_db_connection()
    try:
        connection.execute("UPDATE users SET name=%s WHERE id=%s",(name,session.get("user_id"))); connection.commit(); session["user_name"]=name; flash("Profile updated successfully.","success")
    finally: connection.close()
    return redirect(url_for("profile"))

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

def get_college_from_source_index(exam,source_index):
    exam=str(exam or "").strip().upper(); dataframe=kcet_df if exam=="KCET" else eamcet_df if exam=="EAMCET" else None
    if dataframe is None: return None
    try: source_index=int(source_index)
    except (TypeError,ValueError): return None
    if not (0<=source_index<len(dataframe)): return None
    college=dataframe.iloc[source_index].to_dict(); college["Exam"]=exam; college["SourceIndex"]=source_index; college["Original Index"]=source_index
    college["Website"]=clean_website(college.get("Website",""),college.get("College Name","")); college["Image"]=clean_image(college.get("Image",""))
    college["Exam Fees"]=clean_numeric(college.get("Kcet Fees" if exam=="KCET" else "Eamcet Fees",0))
    return college

@app.route("/saved-colleges")
@login_required
def saved_colleges():
    connection=get_db_connection(); rows=connection.execute("SELECT exam,college_index,created_at FROM saved_colleges WHERE user_id=%s ORDER BY created_at DESC",(session.get("user_id"),)).fetchall(); connection.close()
    colleges=[]
    for row in rows:
        college=get_college_from_source_index(row["exam"],row["college_index"])
        if college is not None: colleges.append(college)
    return render_template("saved_colleges.html",saved_colleges=colleges,colleges=colleges)

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
    # GET FORM DATA
    # --------------------------------------------------------

    exam = (
        request.form.get("exam") or ""
    ).strip().lower()

    index_text = (
        request.form.get("index") or ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATE EXAM
    # --------------------------------------------------------

    if exam not in ALLOWED_EXAMS:

        flash(
            "Invalid exam.",
            "error"
        )

        return redirect(
            request.referrer
            or url_for("colleges")
        )

    # --------------------------------------------------------
    # VALIDATE COLLEGE INDEX
    # --------------------------------------------------------

    try:

        college_index = int(
            index_text
        )

    except (
        ValueError,
        TypeError
    ):

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

        flash(
            "College could not be found.",
            "error"
        )

        return redirect(
            request.referrer
            or url_for("colleges")
        )

    # --------------------------------------------------------
    # DATABASE CONNECTION
    # --------------------------------------------------------

    connection = get_db_connection()

    try:

        # ----------------------------------------------------
        # SAVE COLLEGE
        #
        # PostgreSQL version of SQLite
        # INSERT OR IGNORE
        # ----------------------------------------------------

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
            ),
        )

        # ----------------------------------------------------
        # COMMIT
        # ----------------------------------------------------

        connection.commit()

        # ----------------------------------------------------
        # SUCCESS MESSAGE
        # ----------------------------------------------------

        flash(
            "College saved successfully.",
            "success"
        )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        app.logger.info(
            "College saved | user=%s | exam=%s | index=%s | name=%s",
            session.get("user_id"),
            exam,
            college_index,
            college.get(
                "College Name",
                ""
            ),
        )

    except Exception as error:

        # ----------------------------------------------------
        # ROLLBACK
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CLOSE DATABASE CONNECTION
        # ----------------------------------------------------

        connection.close()

    # --------------------------------------------------------
    # RETURN TO PREVIOUS PAGE
    # --------------------------------------------------------

    return redirect(
        request.referrer
        or url_for("saved_colleges")
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


@app.route("/compare", endpoint="compare")
@app.route("/compare", endpoint="compare_colleges")
@login_required
def compare():
    """Compare colleges selected on the Colleges page.

    The Colleges page sends values such as:
        ?college=kcet:1&college=kcet:5&college=kcet:8

    Older versions sometimes sent plain numeric indices plus an exam query
    parameter. Both formats are supported here.
    """

    selected_values = request.args.getlist("college")

    if not selected_values:
        selected_values = request.args.getlist("indices")

    default_exam = request.args.get("exam", "KCET").strip().upper()
    if default_exam not in ALLOWED_EXAMS:
        default_exam = "KCET"

    colleges = []
    seen = set()

    for value in selected_values:
        value = str(value or "").strip()
        if not value:
            continue

        exam = default_exam
        index_text = value

        # Current JavaScript sends exam:index, e.g. kcet:1.
        if ":" in value:
            possible_exam, possible_index = value.split(":", 1)
            possible_exam = possible_exam.strip().upper()
            if possible_exam in ALLOWED_EXAMS:
                exam = possible_exam
                index_text = possible_index.strip()

        try:
            college_index = int(index_text)
        except (TypeError, ValueError):
            continue

        key = (exam, college_index)
        if key in seen:
            continue
        seen.add(key)

        college = get_college_from_source_index(
            exam,
            college_index,
        )

        if college is not None:
            colleges.append(college)

        if len(colleges) >= 4:
            break

    return render_template(
        "compare.html",
        colleges=colleges,
        saved_colleges=colleges,
        selected_exam=default_exam,
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
def request_too_large(error): return "Request is too large.",413

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Unhandled server error")
    return "An unexpected server error occurred.",500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug=True)



