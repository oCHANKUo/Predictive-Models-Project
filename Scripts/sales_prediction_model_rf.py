from flask import Flask, request, jsonify
from flask_cors import CORS
import pyodbc
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
RF_MODEL_FILE = os.path.join(BASE_DIR, "../models/sales_rf_model.pkl")
RF_SCALER_FILE = os.path.join(BASE_DIR, "../scaler/sales_rf_scaler.pkl")
RF_X_COLUMNS_FILE = os.path.join(BASE_DIR, "../columns/sales_rf_X_columns.pkl")

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

@app.route('/train_sales_rf', methods=['POST', 'GET'])
def train_rf_model():
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

    # Random Forest model
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_scaled, y)

    # Save the RF model, scaler, and columns
    joblib.dump(rf_model, RF_MODEL_FILE)
    joblib.dump(scaler, RF_SCALER_FILE)
    joblib.dump(X.columns, RF_X_COLUMNS_FILE)

    return jsonify({"message": "Random Forest Sales Prediction model trained successfully"})

@app.route('/predict_sales_rf', methods=['GET', 'POST'])
def predict_sales_rf():
    selected_year = request.args.get("year", default=2015, type=int)
    selected_month = request.args.get("month", default=None, type=int)

    if selected_year is None:
        return jsonify({"error": "Year is required"}), 400

    if selected_month is None:
        months_to_predict = range(1, 13)
    else:
        months_to_predict = [selected_month]

    model = joblib.load(RF_MODEL_FILE)
    scaler = joblib.load(RF_SCALER_FILE)
    X_columns = joblib.load(RF_X_COLUMNS_FILE)

    df = fetch_data()
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    min_year = df['Year'].min()

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
            "Month": m,
            "Quarter": future.iloc[0]['Quarter'],
            "PredictedSales": round(float(preds[0]), 2)
        })

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
