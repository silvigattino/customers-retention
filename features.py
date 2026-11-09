#  Feature engineering
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]

NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "n_protection"]
CAT_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod"]
				

def engineer_features(df):
    """Remove duplicated 'No ... service' levels and add n_protection."""
    df = df.copy()
    df[INTERNET_ADDONS] = df[INTERNET_ADDONS].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    df["n_protection"] = (df[PROTECTION_COLS] == "Yes").sum(axis=1)
    return df