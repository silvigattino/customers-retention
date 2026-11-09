USE ChurnDB;
GO

-- Cleaning rules in ONE place: returns customers in the API input contract format.
-- Must match the cleaning done before training (customersretention.ipynb).
CREATE OR ALTER VIEW dbo.vw_CustomerFeatures AS
SELECT
    customerID,
    gender,
    CASE SeniorCitizen WHEN 1 THEN 'Yes' ELSE 'No' END AS SeniorCitizen,  -- 0/1 -> No/Yes
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    CAST(MonthlyCharges AS FLOAT)            AS MonthlyCharges,
    CAST(ISNULL(TotalCharges, 0) AS FLOAT)   AS TotalCharges,          -- NULL -> 0 (new customers)
    Churn
FROM dbo.Customers;
GO

-- One customer, ready to be scored
CREATE OR ALTER PROCEDURE dbo.usp_GetCustomerFeatures
    @CustomerID VARCHAR(20)
AS
BEGIN
    SET NOCOUNT ON;
    SELECT customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
           PhoneService, MultipleLines, InternetService, OnlineSecurity,
           OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
           StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
           MonthlyCharges, TotalCharges
    FROM dbo.vw_CustomerFeatures
    WHERE customerID = @CustomerID;
END;
GO

-- All active customers (not churned yet): the population to score
CREATE OR ALTER PROCEDURE dbo.usp_GetCustomersForScoring
AS
BEGIN
    SET NOCOUNT ON;
    SELECT customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
           PhoneService, MultipleLines, InternetService, OnlineSecurity,
           OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
           StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
           MonthlyCharges, TotalCharges
    FROM dbo.vw_CustomerFeatures
    WHERE Churn = 'No';
END;
GO