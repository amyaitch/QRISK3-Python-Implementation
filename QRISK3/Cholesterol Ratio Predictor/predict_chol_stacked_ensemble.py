import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Configuration ---
CHOLESTEROL_RATIO_FILE = '/Users/amyaitchison/Desktop/MSc/Project/GitHub/output_cholesterol_ratio.csv'
BMI_FILE = '/Users/amyaitchison/Desktop/MSc/Project/GitHub/output_BMI.csv'
LINK_ID_COL = 'LinkId'
VALUE_COL = 'valueNumeric'
BMI_LABEL = 'BMI'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Load and Prepare Data ---
def load_bmi_data(filepath, label):
    """Loads, cleans, and aggregates the BMI data."""
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    df.columns = [col.strip() for col in df.columns]

    if LINK_ID_COL not in df.columns or VALUE_COL not in df.columns or 'DataItemLabel' not in df.columns:
        print(f"Error: Required columns not found in {filepath}")
        return None

    df = df[df['DataItemLabel'] == label]
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors='coerce')
    df.dropna(subset=[LINK_ID_COL, VALUE_COL], inplace=True)

    agg_df = df.groupby(LINK_ID_COL)[VALUE_COL].mean().reset_index()
    return agg_df

print("Loading Cholesterol Ratio data...")
try:
    ratio_df = pd.read_csv(CHOLESTEROL_RATIO_FILE)
except FileNotFoundError:
    print(f"Error: {CHOLESTEROL_RATIO_FILE} not found.")
    exit()

print("Loading and preparing BMI data...")
bmi_df = load_bmi_data(BMI_FILE, BMI_LABEL)

if bmi_df is None:
    print("Failed to load or prepare the BMI dataset. Exiting.")
    exit()

# --- Merge Data ---
print("Merging Cholesterol Ratio and BMI data...")
merged_df = pd.merge(ratio_df, bmi_df, on=LINK_ID_COL)

if merged_df.empty:
    print("Error: Merging resulted in an empty dataframe. Check for common LinkIds.")
    exit()

# --- Prepare for Modeling ---
X = merged_df[[VALUE_COL]] # BMI values
y = merged_df['CholesterolRatio'] # Cholesterol Ratio values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# --- Train Base Models and Get Predictions ---
print("Training base models and generating predictions...")

# Tuned XGBoost Model
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

# --- Create Meta-Features for Meta-Model ---
# Stack predictions from base models as new features
meta_features = np.column_stack((xgb_predictions, rf_predictions, lr_predictions, br_predictions))

# --- Train Meta-Model ---
print("Training meta-model...")
meta_model = LinearRegression() # Using Linear Regression as the meta-model
meta_model.fit(meta_features, y_test) # Train meta-model on predictions and true labels

# --- Make Final Predictions with Stacked Ensemble ---
print("Making final predictions with stacked ensemble...")
stacked_predictions = meta_model.predict(meta_features)

# --- Evaluate Stacked Ensemble ---
mse = mean_squared_error(y_test, stacked_predictions)
rmse = np.sqrt(mse)

print(f"\n--- Stacked Ensemble Model Evaluation Results ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("-------------------------------------------------")
