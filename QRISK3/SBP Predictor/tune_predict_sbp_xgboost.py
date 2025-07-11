import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Configuration ---
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')

SBP_FILE = os.path.join(DATA_DIR, 'output_SBP.csv')
BMI_FILE = os.path.join(DATA_DIR, 'output_BMI.csv')

LINK_ID_COL = 'LinkId'
VALUE_COL = 'valueNumeric'
SBP_LABEL = 'SystolicBloodPressure'
BMI_LABEL = 'BMI'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Load and Prepare Data ---
def load_and_prepare_data(filepath, label, skip=0):
    """Loads a CSV, skips initial rows if needed, and prepares the data."""
    try:
        df = pd.read_csv(filepath, skiprows=skip, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    df.columns = [col.strip() for col in df.columns]

    if LINK_ID_COL not in df.columns or VALUE_COL not in df.columns:
        print(f"Error: Required columns not found in {filepath}")
        return None

    df = df[df['DataItemLabel'] == label]
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors='coerce')
    df.dropna(subset=[LINK_ID_COL, VALUE_COL], inplace=True)

    agg_df = df.groupby(LINK_ID_COL)[VALUE_COL].mean().reset_index()
    return agg_df

print("Loading and preparing data...")
sbp_df = load_and_prepare_data(SBP_FILE, SBP_LABEL, skip=1)
bmi_df = load_and_prepare_data(BMI_FILE, BMI_LABEL)

if sbp_df is None or bmi_df is None:
    print("Failed to load or prepare one or both datasets. Exiting.")
    exit()

# --- Merge Data ---
print("Merging data...")
merged_df = pd.merge(sbp_df, bmi_df, on=LINK_ID_COL, suffixes=['_sbp', '_bmi'])

if merged_df.empty:
    print("Error: Merging resulted in an empty dataframe.")
    exit()

# --- Model Tuning ---
X = merged_df[['valueNumeric_bmi']]
y = merged_df['valueNumeric_sbp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print("Starting hyperparameter tuning with GridSearchCV...")

# Define the grid of parameters to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

# Create the XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    cv=3, # 3-fold cross-validation
    scoring='neg_mean_squared_error', # Use MSE for scoring
    verbose=1, 
    n_jobs=-1 # Use all available CPU cores
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

print("Hyperparameter tuning complete.")

# --- Evaluation ---
print("Evaluating the best model...")

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"\n--- Best Model Evaluation Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("-------------------------------------")
