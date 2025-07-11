import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')

CHOLESTEROL_RATIO_FILE = os.path.join(DATA_DIR, 'output_cholesterol_ratio.csv')
BMI_FILE = os.path.join(DATA_DIR, 'output_BMI.csv')

LINK_ID_COL = 'LinkId'
VALUE_COL = 'valueNumeric'
BMI_LABEL = 'BMI'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Load and Prepare BMI Data ---
def load_bmi_data(filepath, label):
    """Loads, cleans, and aggregates the BMI data."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [col.strip() for col in df.columns]

    required_cols = {LINK_ID_COL, VALUE_COL, 'DataItemLabel'}
    if not required_cols.issubset(df.columns):
        print(f"Error: Required columns not found in {filepath}")
        return None

    df = df[df['DataItemLabel'] == label]
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors='coerce')
    df.dropna(subset=[LINK_ID_COL, VALUE_COL], inplace=True)

    agg_df = df.groupby(LINK_ID_COL)[VALUE_COL].mean().reset_index()
    return agg_df

def main():
    print("Loading Cholesterol Ratio data...")
    if not os.path.exists(CHOLESTEROL_RATIO_FILE):
        print(f"Error: {CHOLESTEROL_RATIO_FILE} not found.")
        return

    ratio_df = pd.read_csv(CHOLESTEROL_RATIO_FILE)

    print("Loading and preparing BMI data...")
    bmi_df = load_bmi_data(BMI_FILE, BMI_LABEL)
    if bmi_df is None:
        print("Failed to load or prepare the BMI dataset. Exiting.")
        return

    print("Merging Cholesterol Ratio and BMI data...")
    merged_df = pd.merge(ratio_df, bmi_df, on=LINK_ID_COL)
    if merged_df.empty:
        print("Error: Merging resulted in an empty dataframe. Check for common LinkIds.")
        return

    # Prepare features and target
    X = merged_df[[VALUE_COL]]  # BMI values
    y = merged_df['CholesterolRatio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Training base models and generating predictions...")

    # XGBoost Model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    # Random Forest Model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    # Bayesian Ridge Model
    br_model = BayesianRidge()
    br_model.fit(X_train, y_train)
    br_predictions = br_model.predict(X_test)

    # Create meta-features from base model predictions
    meta_features = np.column_stack((xgb_predictions, rf_predictions, lr_predictions, br_predictions))

    print("Training meta-model...")
    meta_model = LinearRegression()
    meta_model.fit(meta_features, y_test)

    print("Making final predictions with stacked ensemble...")
    stacked_predictions = meta_model.predict(meta_features)

    mse = mean_squared_error(y_test, stacked_predictions)
    rmse = np.sqrt(mse)

    print(f"\n--- Stacked Ensemble Model Evaluation Results ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()

