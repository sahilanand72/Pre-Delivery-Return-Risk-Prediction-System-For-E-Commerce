import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_target(df):
    """
    Create target variable 'returned' from Return_Status column.
    """

    if "Return_Status" in df.columns:
        df["returned"] = df["Return_Status"].map({
            "Returned": 1,
            "Not Returned": 0
        })
    elif "return_status" in df.columns:
        df["returned"] = df["return_status"].map({
            "Returned": 1,
            "Not Returned": 0
        })
    elif "is_returned" in df.columns:
        df["returned"] = df["is_returned"]
    else:
        raise ValueError(
            f"No return label column found. Available columns: {list(df.columns)}"
        )

    return df

def remove_leakage(df):
    """
    Remove post-delivery leakage columns.
    """

    leakage_cols = [
        "Return_Date",
        "Return_Reason",
        "Days_to_Return"
    ]

    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Numeric → median
    - Categorical → mode
    """
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def encode_categorical_features(X: pd.DataFrame):
    """
    Label encode categorical columns.
    """
    label_encoders = {}
    categorical_cols = X.select_dtypes(include='object').columns

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    return X, label_encoders
