import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.markdown(
    """
    <div style='background-color: #1f2a38; padding: 0px; text-align: center; font-size: 32px; font-weight: bold; color: white; text-shadow: 0 0 10px #FF0000, 0 0 20px #FF0000, 0 0 30px #FF0000;'>
        WTI Crude Oil Price Prediction
    </div>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataase.csv')
    except FileNotFoundError:
        try:
            df = pd.read_excel('dataase.xlsx')
        except FileNotFoundError:
            return None
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    numeric_columns = ['DJIA', 'NASDAQ', 'sp500', 'bitcoin', 'gold', 'Silver']
    for col in numeric_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
            
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def add_features(df_input, target_col='crude oil ( WTI)'):
    df_features = df_input.copy()
    
    # Lag features
    for lag in [1, 3, 5, 7]:
        df_features[f'wti_lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Rolling features
    for window in [7, 14, 30]:
        df_features[f'wti_rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
        df_features[f'wti_rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
            
    df_features['wti_price_change'] = df_features[target_col].pct_change()
    return df_features

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    models = {}
    predictions = {}
    
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    predictions['Linear Regression'] = lr.predict(X_test)
    
    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf.predict(X_test)
    
    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_model.predict(X_test)
    
    # 4. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    predictions['Gradient Boosting'] = gb.predict(X_test)
    
    # 5. SVR
    svr = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
    svr.fit(X_train, y_train)
    models['SVR'] = svr
    predictions['SVR'] = svr.predict(X_test)
    
    return models, predictions

# --- Main App Logic ---

def main():
    st.sidebar.title("Navigation menu")
    page = st.sidebar.radio("", 
        ["Dataset Insights", 
         "Interactive Manual Prediction", 
         "Model Performance & Ranking",
         "Prediction Visualization",
         "Forecast Next 7 Days"]
    )
    st.sidebar.markdown("<br>" * 8, unsafe_allow_html=True)  

    st.sidebar.markdown(
    "<div style='text-align: center;'>Made by<br><b>Farah Belghith & Zaineb Darchem</b></div>",
    unsafe_allow_html=True
    )


    df = load_data()
    if df is None:
        st.error("Dataset not found. Please ensure 'dataase.csv' or 'dataase.xlsx' is in the app directory.")
        return

    target_col = 'crude oil ( WTI)'
    
    # Prepare data for models (Global context)
    df_processed = add_features(df, target_col).dropna()
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models, predictions = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Calculate Results Helper
    results = []
    for name, y_pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R2 Score': r2})
    results_df = pd.DataFrame(results).sort_values(by='RMSE').reset_index(drop=True)
    results_df.index = results_df.index + 1
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    # --- Page 1: Dataset & Overview ---
    if page == "Dataset Insights":
        st.markdown("### Dataset Overview")
        st.markdown("The dataset contains historical daily data for various financial and energy indicators.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Preview")
            st.write(df.head(8))
        with col2:
            st.subheader("Descriptive Statistics")
            st.write(df.describe())

        st.markdown(f"**Target Column:** `{target_col}`")
        st.markdown("We chose West Texas Intermediate (WTI) Crude Oil price as the target variable because it is a key benchmark for oil pricing globally. It is also economically important, has sufficient data, and is easy to interpret.")

    # --- Page 2: Interactive Manual Prediction ---
    elif page == "Interactive Manual Prediction":
        st.header("Interactive Manual Prediction")
        st.markdown("""
        Adjust the sliders below to simulate market conditions and other features. 
        The model will instantly predict the WTI Crude Oil Price based on your inputs.
        """)
        
        # Create a placeholder for the price at the top
        price_placeholder = st.empty()

        # Get top 10 features
        feature_cols = X.columns.tolist()
        top_10_features = feature_cols[:10] 
        
        # Create input dictionary
        user_input = {}
        
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(top_10_features):
            # Determine min/max/mean for slider defaults
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())
            
            with col1 if i % 2 == 0 else col2:
                user_input[feature] = st.slider(
                    f"{feature}", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=mean_val,
                    key=f"slider_{feature}"
                )

        # Fill remaining features with mean values to complete the input vector
        for feature in feature_cols[10:]:
            user_input[feature] = X[feature].mean()
            
        # Calculate Prediction Instantly
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = best_model.predict(input_scaled)[0]
        
        # Display at the top
        price_placeholder.markdown(
            f"""
            <div style="background-color: #e8e8e8; padding: 0px; border-radius: 10px; text-align: center; margin-bottom: 0px;">
                <h3 style="margin:0; color: #31333F;">Predicted WTI Price</h3>
                <h1 style="margin:0; color: #00CC96; font-size: 3em;">${prediction:.2f}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- Page 3: Model Performance ---
    elif page == "Model Performance & Ranking":
        st.header("Model Performance & Ranking")
        
        st.markdown("""
        ### About Tree-Based Models
        Tree-based models like **Random Forest**, **Gradient Boosting**, and **XGBoost** build decisions based on a series of rules (trees). 
        - **Random Forest** builds many independent trees and averages them.
        - **Gradient Boosting & XGBoost** build trees sequentially, where each new tree tries to fix the errors of the previous ones.
        These are often superior for tabular data as they capture non-linear complex relationships well.
        """)
        
        st.table(results_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen').highlight_max(subset=['R2 Score'], color='lightgreen'))
        
        st.subheader("Model Interpretation")
        st.success(f"**Winner: {best_model_name}**")
        st.markdown(f"""
        The **{best_model_name}** model achieved the lowest RMSE.
        
        **Why Gradient Boosting/XGBoost perform well:**
        - They effectively handle complex, non-linear relationships.
        - Robust against noise in financial data.
        
        **Why SVR might perform worse:**
        - SVR can struggle with specific kernels on noisy multi-feature time-series data without extensive tuning.
        """)
        
    # --- Page 4: Prediction Visualization ---
    elif page == "Prediction Visualization":
        st.header("Actual vs Predicted Prices")
        st.markdown("Compare the actual market prices against the model's predictions.")
        
        selected_model_viz = st.selectbox("Select Model to Visualize", results_df['Model'].tolist())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test.values, label='Actual Price', color='black', linewidth=2)
        ax.plot(y_test.index, predictions[selected_model_viz], label=f'Predicted ({selected_model_viz})', color='red', linestyle='--', alpha=0.8)
        ax.set_title(f"Actual vs Predicted ({selected_model_viz})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # --- Page 5: Forecast Next 7 Days ---
    elif page == "Forecast Next 7 Days":
        st.header("Forecast Next 7 Days")
        st.markdown("This section uses the best performing model to forecast WTI prices for the next 7 days, based on the growing trend.")
        
        if st.button("Generate Forecast"):
            # Start with the very last available data point (including raw)
            # We need to simulate the future.
            # Strategy: Take the Full DF. Append 7 new rows with dates.
            # Forward fill exogenous variables (Naive assumption).
            # Iteratively calculate features and predict.
            
            future_days = 7
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
            
            # Create a working copy of the dataframe
            df_forecast = df.copy()
            
            predictions_future = []
            
            progress_bar = st.progress(0)
            
            for i, date in enumerate(future_dates):
                # 1. Append a new row with index = date
                # We use the previous row values for exogenous features (Simple Naive Forecast for inputs)
                last_row = df_forecast.iloc[-1].copy()
                # Set target to NaN effectively (or we can just leave it as last value temporarily, but we want to predict it)
                # Ideally, we don't know the target.
                
                new_row = pd.DataFrame([last_row], index=[date])
                # We interpret 'exogenous' columns as "staying same".
                
                df_forecast = pd.concat([df_forecast, new_row])
                
                # 2. Re-calculate features for this new extended dataframe
                # Note: This is inefficient (re-calcing whole history) but safe and easy for consistency.
                df_features_temp = add_features(df_forecast, target_col)
                
                # 3. Get the feature vector for the *last* row (the one we just added)
                # We need to drop the target col to get X
                last_row_features = df_features_temp.iloc[[-1]].drop(columns=[target_col])
                
                # 4. Scale
                last_row_scaled = scaler.transform(last_row_features)
                
                # 5. Predict
                pred_price = best_model.predict(last_row_scaled)[0]
                predictions_future.append(pred_price)
                
                # 6. Update the DataFrame's target column for this date with the predicted value
                # so that *next* iteration's lag features will use this prediction.
                df_forecast.at[date, target_col] = pred_price
                
                progress_bar.progress((i + 1) / future_days)
            
            # Plotting
            st.subheader("7-Day Forecast Results")
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions_future})
            forecast_df.set_index('Date', inplace=True)
            
            st.table(forecast_df)
            
            # Download Button
            csv = forecast_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Forecast Results as CSV",
                data=csv,
                file_name='7_day_wti_forecast.csv',
                mime='text/csv',
            )
            
            fig_fc, ax_fc = plt.subplots(figsize=(10, 5))
            # Show last 30 days of actual history + 7 days forecast
            history_df = df.iloc[-30:][target_col]
            
            ax_fc.plot(history_df.index, history_df.values, label='Historical (Last 30 Days)', color='black')
            ax_fc.plot(forecast_df.index, forecast_df['Predicted Price'], label='Forecast (Next 7 Days)', color='green', marker='o', linestyle='--')
            
            ax_fc.set_title(f"7-Day Forecast with {best_model_name}")
            ax_fc.set_xlabel("Date")
            ax_fc.set_ylabel("Price (USD)")
            ax_fc.legend()
            ax_fc.grid(True, alpha=0.3)
            st.pyplot(fig_fc)

if __name__ == "__main__":
    main()
