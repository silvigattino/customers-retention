-- Customers table: raw data, as it would come from a CRM
USE ChurnDB;
GO

CREATE TABLE dbo.Customers (
    customerID        VARCHAR(20)   NOT NULL PRIMARY KEY,
    gender            VARCHAR(10)   NOT NULL,
    SeniorCitizen     TINYINT       NOT NULL,   -- raw: 0/1, as in the CRM
    Partner           VARCHAR(3)    NOT NULL,
    Dependents        VARCHAR(3)    NOT NULL,
    tenure            INT           NOT NULL,
    PhoneService      VARCHAR(3)    NOT NULL,
    MultipleLines     VARCHAR(20)   NOT NULL,
    InternetService   VARCHAR(20)   NOT NULL,
    OnlineSecurity    VARCHAR(20)   NOT NULL,
    OnlineBackup      VARCHAR(20)   NOT NULL,
    DeviceProtection  VARCHAR(20)   NOT NULL,
    TechSupport       VARCHAR(20)   NOT NULL,
    StreamingTV       VARCHAR(20)   NOT NULL,
    StreamingMovies   VARCHAR(20)   NOT NULL,
    Contract          VARCHAR(20)   NOT NULL,
    PaperlessBilling  VARCHAR(3)    NOT NULL,
    PaymentMethod     VARCHAR(30)   NOT NULL,
    MonthlyCharges    DECIMAL(8,2)  NOT NULL,
    TotalCharges      DECIMAL(10,2) NULL,       -- raw: NULL for new customers
    Churn VARCHAR(3)    NOT NULL    -- historical label: Yes = already left
);
GO
