-- Analysis queries on the latest predictions (run manually, not part of setup)
USE ChurnDB;
GO

-- Full retention list, highest risk first
EXEC ml.usp_GetRetentionList;

-- Top 20 customers by expected revenue at risk (next 12 months)
SELECT TOP 20 p.customerID,
       ROUND(p.ChurnProbability, 3) AS ChurnProbability,
       CASE p.Contact WHEN 1 THEN 'Contact' ELSE 'No action' END AS Action,
       ROUND(p.ChurnProbability * c.MonthlyCharges * 12, 2) AS RevenueAtRisk12m
FROM ml.ChurnPredictions AS p
JOIN dbo.Customers       AS c ON c.customerID = p.customerID
ORDER BY RevenueAtRisk12m DESC;

-- Summary of each scoring run
SELECT BatchID,
       MIN(ScoredAt)                AS ScoredAt,
       COUNT(*)                     AS Scored,
       SUM(CAST(Contact AS INT))    AS ToContact,
       ROUND(AVG(ChurnProbability), 3) AS AvgProbability
FROM ml.ChurnPredictions
GROUP BY BatchID
ORDER BY ScoredAt DESC;