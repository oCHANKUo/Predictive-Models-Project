from flask import Flask, request, jsonify
import pyodbc
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

app = Flask(__name__)
MODEL_FILE = "sales_prediction_model.pkl"
SCALER_FILE = "scaler.pkl"
X_COLUMNS_FILE = "X_columns.pkl"

# Define save paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
SCALER_FILE = os.path.join(BASE_DIR, "../scaler/scaler.pkl")
X_COLUMNS_FILE = os.path.join(BASE_DIR, "../columns/X_columns.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "../models/sales_model.pkl") 

# Database connection
def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER03;"
        "DATABASE=DataWarehouseClassic;"
        "UID=admin;"
        "PWD=admin"
    )
    return conn

# Fetch data
def fetch_data():
    query = """
    SELECT 
        d.Year,
        d.Month,
        ISNULL(d.Quarter, 0) AS Quarter,
        ISNULL(d.IsHolidaySL, 0) AS IsHolidaySL,
        SUM(f.LineTotal) AS TotalSales
    FROM FactSalesOrderDetail f
    JOIN DimDate d ON f.OrderDateKey = d.DateKey
    GROUP BY d.Year, d.Month, d.Quarter, d.IsHolidaySL
    ORDER BY d.Year, d.Month;
    """
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Train model endpoint
@app.route('/train_sales', methods=['POST', 'GET'])
def train_model():
    df = fetch_data()

    # Ensure correct datatypes
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    df['IsHolidaySL'] = df['IsHolidaySL'].astype(int)
    df['TotalSales'] = df['TotalSales'].astype(float)

    # Continuous month index
    df['MonthIndex'] = (df['Year'] - df['Year'].min()) * 12 + df['Month']

    # Features and target
    X = df[['MonthIndex', 'Month', 'Quarter', 'IsHolidaySL']]
    y = df['TotalSales']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model (Linear Regression, can swap to RandomForest if needed)
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Save model + scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(X.columns, X_COLUMNS_FILE)

    return jsonify({"message": "Monthly Sales Prediction model trained successfully"})


# Train demand
@app.route('/train_demand', methods=['POST', 'GET'])
def train_demand():
    df = fetch_data()

    # target: product quantity 
    y = df['OrderQty']
    numeric_features = ['UnitPrice', 'UnitPriceDiscount']
    x_numeric = df[numeric_features]

    categorical_features = ['UnitPrice', 'UnitPriceDiscount']
    x_categorical = pd.get_dummies(df[categorical_features], drop_first=True)

    df['OrderDateKey'] = pd.to_datetime(df['OrderDateKey'])
    df['OrderYear'] = df['OrderDateKey'].dt.year
    df['OrderMonth'] = df['OrderDateKey'].dt.month
    df['OrderDay'] = df['OrderDateKey'].dt.day
    x_time = df[['OrderYear', 'OrderMonth', 'OrderDay']]

    x = pd.concat([x_numeric, x_categorical, x_time], axis=1).fillna(0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_scaled, y)

    # Save the demand model
    joblib.dump(model, "demand_prediction_model.pkl")
    joblib.dump(scaler, "demand_scaler.pkl")
    joblib.dump(x.columns, "demand_x_columns.pkl")

    return jsonify({"message": "Demand prediction model trained successfully"})

# Prediction endpoint
@app.route('/predict_sales', methods=['GET'])
def predict_sales():
    months_ahead = int(request.args.get("months", 6))  # default = 6

    # Load model + scaler
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    X_columns = joblib.load(X_COLUMNS_FILE)

    # Get historical data
    df = fetch_data()
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)

    # Last known point
    last_year = df['Year'].max()
    last_month = df['Month'].max()
    last_index = ((last_year - df['Year'].min()) * 12 + last_month)

    # Build future dataframe
    future = pd.DataFrame({"MonthIndex": [last_index + i for i in range(1, months_ahead + 1)]})

    # Calculate Year & Month
    future_year_month = []
    for i in range(1, months_ahead + 1):
        month = last_month + i
        year = last_year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        future_year_month.append((year, month))

    future['Year'] = [y for (y, m) in future_year_month]
    future['Month'] = [m for (y, m) in future_year_month]
    future['Quarter'] = ((future['Month'] - 1) // 3 + 1)
    future['IsHolidaySL'] = 0  # assume not holiday

    # Align columns + scale
    X_future = future[X_columns]
    X_future_scaled = scaler.transform(X_future)

    # Predictions
    preds = model.predict(X_future_scaled)

    # Format results
    results = [
        {"Year": int(future.iloc[i]['Year']),
         "Month": int(future.iloc[i]['Month']),
         "Quarter": int(future.iloc[i]['Quarter']),
         "PredictedSales": round(float(preds[i]), 2)}
        for i in range(months_ahead)
    ]

    return jsonify(results)

# Product Demand Predictions
@app.route('/predict_demand', methods=['GET'])
def predict_demand():
    months_ahead = int(request.args.get("months", 6))
    year = request.args.get("year", None)
    month = request.args.get("month", None)

    model = joblib.load("demand_prediction_model.pkl")
    scaler = joblib.load("demand_scaler.pkl")
    x_columns = joblib.load("demand_x_columns.pkl")

    df_train = fetch_data()
    numeric_features = ['UnitPrice', 'UnitPriceDiscount']
    categorical_features = ['ProductKey', 'CustomerKey', 'TerritoryKey', 'SalesPersonKey']

    avg_numeric = df_train[numeric_features].mean().to_dict()
    top_cats = {col: df_train[col].mode()[0] for col in categorical_features}

    df_train['OrderDateKey'] = pd.to_datetime(df_train['OrderDateKey'])
    last_year = df_train['OrderDateKey'].dt.year.max()
    last_month = df_train['OrderDateKey'].dt.month.max()

    if year and month:
        start_year = int(year)
        start_month = int(month)
    else:
        start_year = last_year
        start_month = last_month

    future_dates = []
    for i in range(1, months_ahead + 1):
        month_num = start_month + i
        year_num = ((month_num -1) % 12 ) + 1
        future_dates.append((year_num, month_num))
    
    x_future_rows = []
    for y_val, m_val in future_dates:
        x_numertic_df = pd.DataFrame([avg_numeric])
        x_cateogrical_df = pd.DataFrame({f"{col}_{val}": [1] for col, val in top_cats.items()})
        x_time_df = pd.DataFrame({'OrderYear': [y_val], 'OrderMonth': [m_val], 'OrderDay': [1]})
        x_row = pd.concat([x_numertic_df, x_cateogrical_df, x_time_df], axis=1).fillna(0)
        for col in x_columns:
            if col not in x_row.columns:
                x_row[col] = 0
        x_row = x_row[x_columns]
        x_future_rows.append(x_row)

    x_future = pd.concat(x_future_rows, ignore_index=True)
    x_scaled = scaler.transform(x_future)
    preds = model.predict(x_scaled)

    results = [
        {"Year": int(y_val), "Month": int(m_val), "PredictedDemand": float(round(preds[i], 2))}
        for i, (y_val, m_val) in enumerate(future_dates)
    ]

    return results

# Run app
if __name__ == '__main__':
    app.run(debug=True)
