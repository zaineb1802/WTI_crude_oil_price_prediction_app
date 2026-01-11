# WTI Crude Oil Price Prediction

## Project Overview

This project is a Streamlit web application that predicts the price of West Texas Intermediate (WTI) Crude Oil using historical financial and market indicators. It leverages machine learning models to provide insights, forecasts, and interactive predictions, helping users analyze trends and make informed decisions.

## Features

- **Dataset Insights**: Explore historical data, preview samples, and view descriptive statistics.

- **Interactive Manual Prediction**: Simulate market conditions using sliders to see real-time WTI price predictions.

- **Model Performance & Ranking**: Compare multiple models (Linear Regression, Random Forest, Gradient Boosting, XGBoost, SVR) and select the best-performing one.

- **Prediction Visualization**: Visualize actual vs predicted prices using interactive plots.

- **7-Day Forecast**: Generate and download short-term WTI price forecasts using the best-performing model.


## Technologies Used

- **Python & Streamlit**: Web app and interactive visualization.

- **Pandas & NumPy**: Data handling and preprocessing.

- **Matplotlib**: Plotting and visualization.

- **Scikit-learn**: Machine learning models and metrics.

- **XGBoost**: Gradient boosting model for tabular prediction.

- **Joblib**: Model and scaler serialization.

## Project Structure
```
├── app.py                  
├── train_model.py         
├── dataase.csv / dataase.xlsx  
├── wti_model.pkl          
├── wti_scaler.pkl          
└── README.md               
```

## Installation & App Running 

1. Clone the repository
2. Install required packages
3. Place your dataset in the project directory.
4. Run the Streamlit app

## Usage

Navigate through the sidebar to explore dataset insights, manually adjust features for predictions, view model performance, visualize predictions or forecast WTI prices for the next 7 days and download the results as CSV for further analysis.

## Authors

Farah Belghith & Zaineb Darchem
