"""Feature engineering shared by the notebook and the API."""
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


def engineer_features(df):
    """Remove duplicated 'No ... service' levels and add n_protection."""
    df = df.copy()
    df[INTERNET_ADDONS] = df[INTERNET_ADDONS].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    df["n_protection"] = (df[PROTECTION_COLS] == "Yes").sum(axis=1)
    return df
