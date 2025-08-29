from flask import Flask, request, jsonify
from flask_cors import CORS
import pyodbc
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "rf_sales_model.pkl")
COLUMNS_FILE = os.path.join(BASE_DIR, "rf_sales_columns.pkl")


def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER03;"
        "DATABASE=DataWarehouseClassic;"
        "UID=admin;"
        "PWD=admin"
    )


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


@app.route("/train_sales_rf", methods=['GET', 'POST'])
def train_sales_rf():
    df = fetch_data()

    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    df['IsHolidaySL'] = df['IsHolidaySL'].astype(int)
    df['TotalSales'] = df['TotalSales'].astype(float)
    df['MonthIndex'] = (df['Year'] - df['Year'].min()) * 12 + df['Month']
    
    X = df[['Year', 'MonthIndex', 'Month', 'Quarter', 'IsHolidaySL',
            'TotalOrders', 'AvgUnitPrice', 'AvgDiscount', 'UniqueCustomers', 'AvgShippingTime']]
    y = df['TotalSales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(X.columns.tolist(), COLUMNS_FILE)

    return jsonify({"message": "Random Forest sales model trained and saved."})


@app.route("/predict_sales_rf", methods=['POST', 'GET'])
def predict_sales_rf():
    selected_year = request.args.get("year", default=2015, type=int)
    selected_month = request.args.get("month", default=None, type=int)

    if selected_year is None:
        return jsonify({"error": "Year is required"}), 400

    # Decide which months to predict
    months_to_predict = [selected_month] if selected_month else list(range(1, 13))

    # Ensure no duplicates
    months_to_predict = list(dict.fromkeys(months_to_predict))

    # Load model and feature columns
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(COLUMNS_FILE)

    # Fetch historical data
    df = fetch_data()
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    min_year = df['Year'].min()

    # Compute historical averages for features
    historical_avg = (
        df.groupby('Month', as_index=False).agg({
            'TotalOrders':'mean',
            'AvgUnitPrice':'mean',
            'AvgDiscount':'mean',
            'UniqueCustomers':'mean',
            'AvgShippingTime':'mean'
        }).sort_values('Month')
    )

    months_to_predict = sorted(months_to_predict)

    results = []

    for m in months_to_predict:
        month_index = (selected_year - min_year) * 12 + m

        # Get historical values for month m
        row = historical_avg[historical_avg['Month']==m].iloc[0]

        # Construct a single-row DataFrame with all features
        future = pd.DataFrame([{
            "MonthIndex": month_index,
            "Year": selected_year,
            "Month": m,
            "Quarter": (m - 1)//3 + 1,
            "IsHolidaySL": 0,
            "TotalOrders": row['TotalOrders'],
            "AvgUnitPrice": row['AvgUnitPrice'],
            "AvgDiscount": row['AvgDiscount'],
            "UniqueCustomers": row['UniqueCustomers'],
            "AvgShippingTime": row['AvgShippingTime']
        }])

        X_future = future[feature_columns]
        preds = model.predict(X_future)

        results.append({
            "Year": selected_year,
            "Month": m,
            "Quarter": future.iloc[0]['Quarter'],
            "PredictedSales": round(float(preds[0]), 2)
        })

    results = sorted(results, key=lambda x: x["Month"])

    return jsonify(results)


if __name__ == "__main__":
    app.run(port=5004, debug=True)
