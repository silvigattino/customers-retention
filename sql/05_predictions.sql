USE ChurnDB;
GO

-- Separate schema for model outputs
IF SCHEMA_ID('ml') IS NULL
    EXEC('CREATE SCHEMA ml');
GO

-- One row per customer per scoring run
IF OBJECT_ID('ml.ChurnPredictions') IS NULL
CREATE TABLE ml.ChurnPredictions (
    PredictionID      INT IDENTITY(1,1) PRIMARY KEY,
    BatchID           UNIQUEIDENTIFIER NOT NULL,           -- groups one scoring run
    customerID        VARCHAR(20)      NOT NULL,
    ChurnProbability  FLOAT            NOT NULL,
    Contact           BIT              NOT NULL,
    Threshold         FLOAT            NOT NULL,
    ModelName         VARCHAR(50)      NOT NULL,
    ModelCreated      DATE             NOT NULL,           -- which trained model produced it
    ScoredAt          DATETIME2(0)     NOT NULL DEFAULT SYSUTCDATETIME()   -- UTC
);
GO

-- The API can insert predictions, but not read, update or delete them
GRANT INSERT ON ml.ChurnPredictions TO api_user_ChurnDB;
GO

-- Retention list for marketing: customers to contact in the latest run, highest risk first
CREATE OR ALTER PROCEDURE ml.usp_GetRetentionList
AS
BEGIN
    SET NOCOUNT ON;
    DECLARE @LastBatch UNIQUEIDENTIFIER =
        (SELECT TOP 1 BatchID FROM ml.ChurnPredictions ORDER BY ScoredAt DESC);

    SELECT ROW_NUMBER() OVER (ORDER BY p.ChurnProbability DESC)  AS RiskRank,
           p.customerID,
           ROUND(p.ChurnProbability, 3)                            AS ChurnProbability,
           CASE p.Contact WHEN 1 THEN 'Contact' ELSE 'No action' END AS Action,
           ROUND(p.ChurnProbability * c.MonthlyCharges * 12, 2)    AS RevenueAtRisk12m,
           c.Contract,
           c.tenure,
           c.MonthlyCharges,
           c.PaymentMethod,
           p.ScoredAt
    FROM ml.ChurnPredictions AS p
    JOIN dbo.Customers       AS c ON c.customerID = p.customerID
    WHERE p.BatchID = @LastBatch
    ORDER BY p.ChurnProbability DESC;
END;
GO