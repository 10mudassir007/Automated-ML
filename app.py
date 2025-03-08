import numpy as np
import modin.pandas as pd
import fireducks.pandas as pd
import pickle
import json
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import r2_score, accuracy_score

def remove_unique(df, target_col=None):
    df = df.copy()  
    if target_col is None:
        unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    else:
        unique_cols = [col for col in df.columns if col != target_col and df[col].nunique() == len(df)]
    if len(df.columns) - len(unique_cols) < 5:
        return df
    df.drop(columns=unique_cols, inplace=True)
    return df

def handle_nulls(df, threshold=0.1):
    df = df.copy()
    total_missing = df.isnull().sum().sum() / df.size

    if total_missing <= threshold:
        df = df.dropna()
    else:
        df = df.fillna(df.median(numeric_only=True))
    return df

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

            new_col_name = f"{col1}_div_{col2}"
            df[new_col_name] = (df[col1] + 1e-5) / ((df[col2] + 1e-5) + 1)

            min_val = df[new_col_name].min()
            max_val = df[new_col_name].max()

            if max_val != min_val:  
                df[new_col_name] = 2 * ((df[new_col_name] - min_val) / (max_val - min_val)) - 1
            else:
                df[new_col_name] = 0  

            new_features += 1
            feature_pairs.append((col1, col2))

    return df, feature_pairs

def save_processing_artifacts(feature_pairs, encoders, scalers, filename="processing_artifacts.json"):
    artifacts = {
        "feature_pairs": feature_pairs,
        "encoders": {col: enc.classes_.tolist() for col, enc in encoders.items()},
        "scalers": list(scalers.keys())
    }
    
    with open(filename, "w") as f:
        json.dump(artifacts, f)

    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

def seperate_data(df,target=None):
    if target:
        X = df.drop(target,axis=1)
        y = df[target]
        return train_test_split(X,y,test_size=0.2,random_state=42)
    else:
        return train_test_split(df,test_size=0.2,random_state=42)

def process_df(df, target=None):
    df = df.copy()
    df = remove_unique(df, target)
    df = handle_nulls(df)
    df, scalers = auto_scale(df, target)
    df, encoders = encode_or_drop(df, target)
    df, feature_pairs = feature_engineering(df, target_col=target)
    save_processing_artifacts(feature_pairs, encoders, scalers)
    return df,scalers,encoders,feature_pairs

def training_s(df, target):
    X_train, X_test, y_train, y_test = seperate_data(df, target)
    is_regression = df[target].nunique() > int(df.shape[0] * 0.01)
    models = {}
    if is_regression:
        models = {
            "XGBRegressor": XGBRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=0),
            "LGBMRegressor": LGBMRegressor(verbose=-1)
        }
        scoring_func = r2_score  
    else:
        models = {
            "XGBClassifier": XGBClassifier(),
            "CatBoostClassifier": CatBoostClassifier(verbose=0),
            "LGBMClassifier": LGBMClassifier(verbose=-1)
        }
        scoring_func = accuracy_score 

    best_model = None
    best_score = float('-inf')
    for _, model in models.items():
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        train_score = scoring_func(y_train.to_numpy(), model.predict(X_train.to_numpy()))
        test_score = scoring_func(y_test.to_numpy(), model.predict(X_test.to_numpy()))
        mean_score = (train_score + test_score) / 2  

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
    return best_model,(best_model.score(X_train.to_numpy(),y_train.to_numpy()), best_model.score(X_test.to_numpy(),y_test.to_numpy()))

def load_processing_artifacts(artifacts_file="processing_artifacts.json", scalers_file="scalers.pkl"):
    with open(artifacts_file, "r") as f:
        artifacts = json.load(f)
    
    with open(scalers_file, "rb") as f:
        scalers = pickle.load(f)

    encoders = {}
    for col, classes in artifacts["encoders"].items():
        le = LabelEncoder()
        le.classes_ = np.array(classes)  # Load previously fitted classes
        encoders[col] = le
        
    return artifacts["feature_pairs"], encoders, scalers

def process_user_input(user_df, encoders, scalers, feature_pairs):
    user_df = user_df.copy()

    # Apply Encoding
    for col, encoder in encoders.items():
        if col in user_df:
            user_df[col] = user_df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    # Apply Scaling
    for col, scaler in scalers.items():
        if col in user_df:
            user_df[col] = scaler.transform(user_df[[col]])

    # Apply Feature Engineering
    for col1, col2 in feature_pairs:
        new_col_name = f"{col1}_div_{col2}"
        user_df[new_col_name] = (user_df[col1] + 1e-5) / ((user_df[col2] + 1e-5) + 1)
        
        # Normalize
        min_val, max_val = user_df[new_col_name].min(), user_df[new_col_name].max()
        if max_val != min_val:
            user_df[new_col_name] = 2 * ((user_df[new_col_name] - min_val) / (max_val - min_val)) - 1
        else:
            user_df[new_col_name] =  float(str(1 + (0 * len(str(user_df[new_col_name])))))

    return user_df


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df_uploaded = df  # Save to session state
    st.success("Dataset uploaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()  # Stop execution if no file is uploaded

df = st.session_state.df_uploaded[:250000]

# Ensure the dataset has enough columns
if df.shape[1] < 2:
    st.error("Dataset must have at least one feature and a target column.")
    st.stop()

st.write("Data:")
st.write(df)

with st.spinner("Processing Data"):
    df_processed,scalers,encoders,feature_pairs = process_df(df, df.columns[-1])
    st.success("Data Processed")
if "trained_model" not in st.session_state or "train_scores" not in st.session_state:
    
    with st.spinner("Training the model"):
        model, scores = training_s(df_processed, df.columns[-1])
    st.success("Model Trained ")

    st.session_state.trained_model = model
    st.session_state.train_scores = scores
    st.session_state.df_processed = df_processed
else:
    model = st.session_state.trained_model
    scores = st.session_state.train_scores
    df_processed = st.session_state.df_processed

st.write(f"##### Train Score: {scores[0]}", unsafe_allow_html=True)
st.write(f"##### Test Score: {scores[1]}", unsafe_allow_html=True)

if "feature_inputs" not in st.session_state:
    st.session_state.feature_inputs = {}

with st.form("input_form"):
    
    for col in df_processed.columns:
        if col == df.columns[-1]:
            break
        st.session_state.feature_inputs[col] = st.text_input(
            f"Enter {col}",
            value=st.session_state.feature_inputs.get(col, "")
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = {col: value for col, value in st.session_state.feature_inputs.items()}
    user_df = pd.DataFrame([user_input]) 
    feature_pairs, encoders, scalers = load_processing_artifacts()
    processed_input = process_user_input(user_df, encoders, scalers, feature_pairs)
    
    prediction = model.predict(processed_input.to_numpy())
    target_col = df.columns[-1]
    is_classification = target_col in encoders

    if is_classification:
        label_encoder = encoders[target_col] 
        prediction = label_encoder.inverse_transform([int(prediction[0])])[0]
        st.write("aa",prediction) 
    else:
        try:
            prediction = round(float(prediction[0]), 2)
        except:
            prediction = prediction

    # Show prediction
    st.write(f"#### Predicted Value: {prediction}", unsafe_allow_html=True)