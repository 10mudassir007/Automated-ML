import numpy as np
import modin.pandas as pd
import pickle
import json
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

# Remove unique columns
def remove_unique(df, target_col=None):
    df = df.copy()  
    if target_col is None:
        unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    else:
        unique_cols = [col for col in df.columns if col != target_col and df[col].nunique() == len(df)]
    df.drop(columns=unique_cols, inplace=True)
    return df

# Handle null values
def handle_nulls(df, threshold=0.1):
    df = df.copy()
    total_missing = df.isnull().sum().sum() / df.size

    if total_missing <= threshold:
        df = df.dropna()
    else:
        df = df.fillna(df.median(numeric_only=True))
    return df

# Encode categorical features or drop them
def encode_or_drop(df, target=None):
    df = df.copy()
    encoders = {}

    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target is not None:
        object_cols = [col for col in object_cols if col != target]  

    for col in object_cols:
        unique_count = df[col].nunique()
        if unique_count <= 100:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
        else:
            df.drop(columns=[col], inplace=True)

    return df, encoders

# Scaling numerical features
def auto_scale(df, target=None):
    df = df.copy()
    scalers = {}

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target is not None:
        num_cols = [col for col in num_cols if col != target]

    for col in num_cols:
        col_data = df[col].dropna()

        if col_data.nunique() == 1:
            continue

        if col_data.min() >= 0 and col_data.max() <= 1:
            continue

        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        outliers = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()

        skewness = col_data.skew()

        if abs(skewness) < 0.5:
            scalers[col] = StandardScaler()
        elif outliers > 0:
            scalers[col] = RobustScaler()
        else:
            scalers[col] = MinMaxScaler()

        df[col] = scalers[col].fit_transform(df[[col]])

    return df, scalers


def feature_engineering(df, target_col):
    df = df.copy()
    feature_pairs = []
    target_correlation = df.corr()[target_col]

    strong_correlations = target_correlation[abs(target_correlation) > 0.4]
    if len(strong_correlations) <= 2:
        strong_correlations = target_correlation[abs(target_correlation) > 0.2]

    strong_correlations = strong_correlations.drop(target_col, errors="ignore")
    correlated_features = strong_correlations.index.tolist()

    new_features = 0
    for i in range(len(correlated_features)):
        for j in range(i + 1, len(correlated_features)):
            if new_features >= 3: 
                return df, feature_pairs

            col1, col2 = correlated_features[i], correlated_features[j]

            df[col1] = df[col1] + 1e-5
            df[col2] = df[col2] + 1e-5

            new_col_name = f"{col1}_div_{col2}"
            df[new_col_name] = df[col1] / (df[col2] + 1)

            min_val = df[new_col_name].min()
            max_val = df[new_col_name].max()

            if max_val != min_val:  
                df[new_col_name] = 2 * ((df[new_col_name] - min_val) / (max_val - min_val)) - 1
            else:
                df[new_col_name] = 0  

            new_features += 1
            feature_pairs.append((col1, col2))

    return df, feature_pairs

def feature_engineering(df, target_col=None):
    df = df.copy()
    feature_pairs = []

    if target_col and target_col in df.columns:
        correlation_matrix = df.corr()[target_col]
        strong_correlations = correlation_matrix[abs(correlation_matrix) > 0.4]

        if len(strong_correlations) <= 2:
            strong_correlations = correlation_matrix[abs(correlation_matrix) > 0.2]

        strong_correlations = strong_correlations.drop(target_col, errors="ignore")
        correlated_features = strong_correlations.index.tolist()
    else:
        correlation_matrix = df.corr()
        correlated_features = correlation_matrix.columns.tolist()

    new_features = 0
    for i in range(len(correlated_features)):
        for j in range(i + 1, len(correlated_features)):
            if new_features >= 3:  
                return df, feature_pairs

            col1, col2 = correlated_features[i], correlated_features[j]

            df[col1] = df[col1] + 1e-5
            df[col2] = df[col2] + 1e-5

            new_col_name = f"{col1}_div_{col2}"
            df[new_col_name] = df[col1] / (df[col2] + 1)

            min_val = df[new_col_name].min()
            max_val = df[new_col_name].max()

            if max_val != min_val:  
                df[new_col_name] = 2 * ((df[new_col_name] - min_val) / (max_val - min_val)) - 1
            else:
                df[new_col_name] = 0  

            new_features += 1
            feature_pairs.append((col1, col2))

    return df, feature_pairs

# Save the processing artifacts
def save_processing_artifacts(feature_pairs, encoders, scalers, filename="processing_artifacts.json"):
    artifacts = {
        "feature_pairs": feature_pairs,
        "encoders": {col: enc.classes_.tolist() for col, enc in encoders.items()},
        "scalers": list(scalers.keys())
    }
    
    # Save as JSON
    with open(filename, "w") as f:
        json.dump(artifacts, f)

    # Save scalers separately using Pickle
    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

# Process the dataframe
def process_df(df, target=None):
    df = df.copy()
    
    df = remove_unique(df, target)
    df = handle_nulls(df)
    df, scalers = auto_scale(df, target)
    df, encoders = encode_or_drop(df, target)
    df, feature_pairs = feature_engineering(df, target_col=target)

    save_processing_artifacts(feature_pairs, encoders, scalers)

    return df

# Streamlit App
df = pd.read_csv("Clean_Dataset.csv")
st.write("Original Data:")
st.write(df)

# Processed Data
df_processed = process_df(df)
st.write("Processed Data:")
st.write(df_processed)
