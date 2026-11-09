-- Creates the database used by the churn API (safe to run more than once)
IF DB_ID('ChurnDB') IS NULL
    CREATE DATABASE ChurnDB;
GO