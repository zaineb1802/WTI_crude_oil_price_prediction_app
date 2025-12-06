import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Configuration
DATA_FILE = 'dataase.csv'
MODEL_FILE = 'wti_model.pkl'
SCALER_FILE = 'wti_scaler.pkl'
TARGET_COL = 'crude oil ( WTI)'

def load_and_process_data():
    if not os.path.exists(DATA_FILE):
        # Fallback to xlsx if csv not found
        if os.path.exists('dataase.xlsx'):
            print("Loading from dataase.xlsx...")
            df = pd.read_excel('dataase.xlsx')
            df.to_csv(DATA_FILE, index=False)
        else:
            raise FileNotFoundError(f"Could not find {DATA_FILE} or dataase.xlsx")
    else:
        print(f"Loading from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)

    # Date processing
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean numeric columns
    numeric_columns = ['DJIA', 'NASDAQ', 'sp500', 'bitcoin', 'gold', 'Silver']
    for col in numeric_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)

    df.set_index('date', inplace=True)
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def feature_engineering(df):
    print("Generating features...")
    df_features = df.copy()
    
    # Lags
    for lag in [1, 3, 5, 7]:
        df_features[f'wti_lag_{lag}'] = df_features[TARGET_COL].shift(lag)
        
    # Rolling stats
    for window in [7, 14, 30]:
        df_features[f'wti_rolling_mean_{window}'] = df_features[TARGET_COL].rolling(window=window).mean()
        df_features[f'wti_rolling_std_{window}'] = df_features[TARGET_COL].rolling(window=window).std()
        
    # Price change
    df_features['wti_price_change'] = df_features[TARGET_COL].pct_change()
    
    # Drop NaN created by lags/rolling
    df_features = df_features.dropna()
    
    return df_features

def train_model():
    # Load Data
    try:
        df = load_and_process_data()
    except FileNotFoundError as e:
        print(e)
        return

    # Feature Engineering
    df_features = feature_engineering(df)
    
    X = df_features.drop(columns=[TARGET_COL])
    y = df_features[TARGET_COL]
    
    # Split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting (Best Model)
    print("Training Gradient Boosting Regressor...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = gb.predict(X_train_scaled)
    test_pred = gb.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Save
    print("Saving model and scaler...")
    joblib.dump(gb, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Done!")

if __name__ == "__main__":
    train_model()
