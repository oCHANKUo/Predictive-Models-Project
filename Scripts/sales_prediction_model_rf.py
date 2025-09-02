from flask import Flask, request, jsonify
from flask_cors import CORS
import pyodbc
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_FILE = "sales_rf_model.pkl"

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
def train_sales_rf():
    try:
        df = fetch_data()
        
        features = ['Year', 'Month', 'Quarter', 'IsHolidaySL', 'TotalOrders', 
                    'AvgUnitPrice', 'AvgDiscount', 'UniqueCustomers', 'AvgShippingTime']
        X = df[features]
        y = df['TotalSales']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        
        joblib.dump(rf, MODEL_FILE)
        
        return jsonify({"message": "Model trained and saved successfully!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_sales_rf', methods=['GET'])
def predict_sales():
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "Model not trained yet. Call /train_sales_rf first."}), 400

    try:
        rf = joblib.load(MODEL_FILE)
        
        selected_year = request.args.get("year", type=int, default=2025)
        selected_month = request.args.get("month", type=int, default=None)

        df = fetch_data()
        avg_values = df[['IsHolidaySL', 'TotalOrders', 'AvgUnitPrice', 
                         'AvgDiscount', 'UniqueCustomers', 'AvgShippingTime']].mean()

        months_to_predict = [selected_month] if selected_month else list(range(1, 13))

        results = []
        for m in months_to_predict:
            quarter = ((m-1)//3) + 1
            X_pred = [[
                selected_year,
                m,
                quarter,
                avg_values['IsHolidaySL'],
                avg_values['TotalOrders'],
                avg_values['AvgUnitPrice'],
                avg_values['AvgDiscount'],
                avg_values['UniqueCustomers'],
                avg_values['AvgShippingTime']
            ]]
            pred = rf.predict(X_pred)
            results.append({
                "Year": selected_year,
                "Month": m,
                "Quarter": quarter,
                "PredictedSales": float(round(pred[0], 2))
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
