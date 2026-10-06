CREATE DATABASE IF NOT EXISTS college_predictor;

USE college_predictor;

-- =========================================
-- USERS TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,

    username VARCHAR(100) NOT NULL UNIQUE,

    email VARCHAR(150) NOT NULL UNIQUE,

    password VARCHAR(255) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================================
-- COLLEGES TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS colleges (
    id INT AUTO_INCREMENT PRIMARY KEY,

    college_name VARCHAR(255) NOT NULL,

    branch VARCHAR(100) NOT NULL,

    cutoff_rank INT,

    rating DECIMAL(2,1),

    location VARCHAR(150),

    management_fees DECIMAL(12,2),

    exam_fees DECIMAL(12,2),

    hostel_fees DECIMAL(12,2),

    state VARCHAR(100),

    exam VARCHAR(50) NOT NULL,

    image VARCHAR(255),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================================
-- SAVED COLLEGES TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS saved_colleges (
    id INT AUTO_INCREMENT PRIMARY KEY,

    user_id INT NOT NULL,

    college_id INT NOT NULL,

    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    FOREIGN KEY (college_id)
        REFERENCES colleges(id)
        ON DELETE CASCADE,

    UNIQUE(user_id, college_id)
);


-- =========================================
-- PREDICTION HISTORY TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS prediction_history (
    id INT AUTO_INCREMENT PRIMARY KEY,

    user_id INT,

    exam VARCHAR(50),

    student_rank INT,

    branch VARCHAR(100),

    category VARCHAR(50),

    gender VARCHAR(50),

    preferred_location VARCHAR(150),

    state VARCHAR(100),

    fee_range VARCHAR(100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL
);


-- =========================================
-- EMAIL VERIFICATIONS / OTP TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS email_verifications (
    id INT AUTO_INCREMENT PRIMARY KEY,

    name VARCHAR(100) NOT NULL,

    email VARCHAR(150) NOT NULL UNIQUE,

    password VARCHAR(255) NOT NULL,

    otp_hash VARCHAR(255) NOT NULL,

    expires_at DATETIME NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);