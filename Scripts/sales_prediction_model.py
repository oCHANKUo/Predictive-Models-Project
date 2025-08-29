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

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER03;"
        "DATABASE=DataWarehouseClassic;"
        "UID=admin;"
        "PWD=admin"
    )
    return conn

def fetch_data():
    query = """
    SELECT 
        d.Year,
        d.Month,
        ISNULL(d.Quarter, 0) AS Quarter,
        ISNULL(d.IsHolidaySL, 0) AS IsHolidaySL,
        SUM(f.LineTotal) AS TotalSales,
        COUNT(DISTINCT f.SalesOrderID) AS TotalOrders,
        AVG(f.UnitPrice) AS AvgUnitPrice,
        AVG(f.UnitPriceDiscount) AS AvgDiscount,
        COUNT(DISTINCT f.CustomerKey) AS UniqueCustomers,
        AVG(DATEDIFF(
            DAY, 
            CAST(CONVERT(VARCHAR(8), f.OrderDateKey) AS DATE), 
            CAST(CONVERT(VARCHAR(8), f.ShipDateKey) AS DATE)
        )) AS AvgShippingTime
    FROM FactSalesOrderDetail f
    JOIN DimDate d ON f.OrderDateKey = d.DateKey
    GROUP BY d.Year, d.Month, d.Quarter, d.IsHolidaySL
    ORDER BY d.Year, d.Month;
    """
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@app.route('/train_sales', methods=['POST', 'GET'])
def train_model():
    df = fetch_data()

    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    df['IsHolidaySL'] = df['IsHolidaySL'].astype(int)
    df['TotalSales'] = df['TotalSales'].astype(float)

    df['MonthIndex'] = (df['Year'] - df['Year'].min()) * 12 + df['Month']

    X = df[['MonthIndex', 'Month', 'Quarter', 'IsHolidaySL',
            'TotalOrders', 'AvgUnitPrice', 'AvgDiscount', 'UniqueCustomers', 'AvgShippingTime']]
    y = df['TotalSales']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(X.columns, X_COLUMNS_FILE)

    return jsonify({"message": "Monthly Sales Prediction model trained successfully"})


@app.route('/predict_sales', methods=['GET', 'POST'])
def predict_sales():
    months_ahead = request.args.get("months", default=6, type=int)
    years_ahead = request.args.get("years", default=0, type=int)

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    X_columns = joblib.load(X_COLUMNS_FILE)

    df = fetch_data()
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)

    last_year = df['Year'].max()
    last_month = df['Month'].max()
    last_index = ((last_year - df['Year'].min()) * 12 + last_month)

    historical_avg = df.groupby('Month').agg({
        'TotalOrders':'mean',
        'AvgUnitPrice':'mean',
        'AvgDiscount':'mean',
        'UniqueCustomers':'mean',
        'AvgShippingTime':'mean'
    }).reset_index()

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
    future['IsHolidaySL'] = 0 

    for col in ['TotalOrders', 'AvgUnitPrice', 'AvgDiscount', 'UniqueCustomers', 'AvgShippingTime']:
        future[col] = future['Month'].apply(lambda m: historical_avg.loc[historical_avg['Month']==m, col].values[0])

    X_future = future[X_columns]
    X_future_scaled = scaler.transform(X_future)

    preds = model.predict(X_future_scaled)

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
