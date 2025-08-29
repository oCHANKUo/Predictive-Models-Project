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


@app.route('/predict_sales', methods=['GET'])
def predict_sales():
    selected_year = request.args.get("year", type=int)
    selected_month = request.args.get("month", default=None, type=int)

    if selected_year is None:
        return jsonify({"error": "Year is required"}), 400

    if selected_month is None:
        months_to_predict = range(1, 13)
    else:
        months_to_predict = [selected_month]

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    X_columns = joblib.load(X_COLUMNS_FILE)

    df = fetch_data()
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    min_year = df['Year'].min()

    month_index = (selected_year - min_year) * 12 + selected_month

    historical_avg = df.groupby('Month').agg({
        'TotalOrders':'mean',
        'AvgUnitPrice':'mean',
        'AvgDiscount':'mean',
        'UniqueCustomers':'mean',
        'AvgShippingTime':'mean'
    }).reset_index()

    results = []

    for m in months_to_predict:
        month_index = (selected_year - min_year) * 12 + m

        future = pd.DataFrame([{
            "MonthIndex": month_index,
            "Year": selected_year,
            "Month": m,
            "Quarter": (m - 1)//3 + 1,
            "IsHolidaySL": 0,
            "TotalOrders": historical_avg.loc[historical_avg['Month']==m, 'TotalOrders'].values[0],
            "AvgUnitPrice": historical_avg.loc[historical_avg['Month']==m, 'AvgUnitPrice'].values[0],
            "AvgDiscount": historical_avg.loc[historical_avg['Month']==m, 'AvgDiscount'].values[0],
            "UniqueCustomers": historical_avg.loc[historical_avg['Month']==m, 'UniqueCustomers'].values[0],
            "AvgShippingTime": historical_avg.loc[historical_avg['Month']==m, 'AvgShippingTime'].values[0]
        }])

        X_future = future[X_columns]
        X_future_scaled = scaler.transform(X_future)
        preds = model.predict(X_future_scaled)

        results.append({
            "Year": selected_year,
            "Month": selected_month,
            "Quarter": future.iloc[0]['Quarter'],
            "PredictedSales": round(float(preds[0]), 2)
        })

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
