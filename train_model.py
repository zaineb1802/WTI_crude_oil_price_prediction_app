import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

DATA_FILE = 'dataase.csv'
MODEL_FILE = 'wti_model.pkl'
SCALER_FILE = 'wti_scaler.pkl'
TARGET_COL = 'crude oil ( WTI)'

def load_and_process_data():
    if not os.path.exists(DATA_FILE):
        if os.path.exists('dataase.xlsx'):
            print("Loading from dataase.xlsx...")
            df = pd.read_excel('dataase.xlsx')
            df.to_csv(DATA_FILE, index=False)
        else:
            raise FileNotFoundError(f"Could not find {DATA_FILE} or dataase.xlsx")
    else:
        print(f"Loading from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)

    df['date'] = pd.to_datetime(df['date'])
    
    numeric_columns = ['DJIA', 'NASDAQ', 'sp500', 'bitcoin', 'gold', 'Silver']
    for col in numeric_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)

    df.set_index('date', inplace=True)
    
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
    
    df_features = df_features.dropna()
    
    return df_features

def train_model():
    try:
        df = load_and_process_data()
    except FileNotFoundError as e:
        print(e)
        return

    df_features = feature_engineering(df)
    
    corr_matrix = df_features.corr()
    target_corr = corr_matrix[TARGET_COL].abs().sort_values(ascending=False)
    
    selected_features = target_corr[target_corr > 0.5].index.tolist()
    if TARGET_COL in selected_features:
        selected_features.remove(TARGET_COL)
        
    print(f"Selected {len(selected_features)} features with correlation > 0.5: {selected_features}")
    
    X = df_features[selected_features]
    y = df_features[TARGET_COL]
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training SVR (Linear)...")
    from sklearn.svm import SVR
    model = SVR(kernel='linear', C=100, gamma='auto', epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    print("Saving model and scaler...")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Done!")

if __name__ == "__main__":
    train_model()
