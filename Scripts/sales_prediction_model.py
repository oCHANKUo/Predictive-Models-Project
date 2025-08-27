from flask import Flask, request, jsonify
from flask_cors import CORS
import pyodbc
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)
MODEL_FILE = "sales_prediction_model.pkl"
SCALER_FILE = "scaler.pkl"
X_COLUMNS_FILE = "X_columns.pkl"

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

    model = LinearRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(X.columns, X_COLUMNS_FILE)

    return jsonify({"message": "Monthly Sales Prediction model trained successfully"})


# Prediction endpoint
@app.route('/predict_sales', methods=['GET'])
def predict_sales():
    months_ahead = int(request.args.get("months", 6))  # default = 6

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    X_columns = joblib.load(X_COLUMNS_FILE)

    df = fetch_data()
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)

    last_year = df['Year'].max()
    last_month = df['Month'].max()
    last_index = ((last_year - df['Year'].min()) * 12 + last_month)

    # Build future dataframe
    future = pd.DataFrame({"MonthIndex": [last_index + i for i in range(1, months_ahead + 1)]})

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

    X_future = future[X_columns]
    X_future_scaled = scaler.transform(X_future)

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

if __name__ == '__main__':
    app.run(debug=True)
