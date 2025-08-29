from flask import Flask, request, jsonify
import os
import pyodbc
import pickle
import pandas as pd
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../scripts
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))    # project root
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
SCALER_DIR  = os.path.join(ROOT_DIR, "scaler")
COLUMNS_DIR = os.path.join(ROOT_DIR, "columns")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(COLUMNS_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODELS_DIR,  "demand_model.pkl")
SCALER_PATH  = os.path.join(SCALER_DIR,  "demand_scaler.pkl")
COLUMNS_PATH = os.path.join(COLUMNS_DIR, "demand_column.pkl")

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER03;"
        "DATABASE=DataWarehouseClassic;"
        "UID=admin;"
        "PWD=admin"
    )
    return conn

def fetch_monthly_demand():
    conn = get_connection()
    query = """
        SELECT 
            d.Year AS Year,
            d.Month AS Month,
            ISNULL(d.Quarter, 0) AS Quarter,
            ISNULL(d.IsHolidaySL, 0) AS IsHolidaySL,
            SUM(f.OrderQty) AS TotalQty,
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
    df = pd.read_sql(query, conn)
    conn.close()

    for col in ["Year", "Month", "Quarter", "IsHolidaySL"]:
        df[col] = df[col].astype(int)
    for col in ["TotalQty", "AvgUnitPrice", "AvgDiscount", "AvgShippingTime"]:
        df[col] = df[col].astype(float)
    for col in ["TotalOrders", "UniqueCustomers"]:
        df[col] = df[col].astype(int)

    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    start_year = df["Year"].min()
    df = df.copy()
    df["MonthIndex"] = (df["Year"] - start_year) * 12 + df["Month"]
    return df

@app.route("/train_demand", methods=["POST", "GET"])
def train_demand():
    df = fetch_monthly_demand()
    df = add_calendar_features(df)

    feature_cols = [
        "Year", "Month", "Quarter", "IsHolidaySL", "MonthIndex",
        "TotalOrders", "AvgUnitPrice", "AvgDiscount", "UniqueCustomers", "AvgShippingTime",
    ]
    X = df[feature_cols]
    y = df["TotalQty"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(feature_cols, f)

    return jsonify({"message": "Demand model trained successfully"})

@app.route("/predict_demand", methods=['GET','POST'])
def predict_demand():
    months_ahead = int(request.args.get("months", 6)) 

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(COLUMNS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    hist = fetch_monthly_demand()
    hist = add_calendar_features(hist)

    last_year = hist.loc[hist["Year"].idxmax(), "Year"]
    last_month = hist[hist["Year"] == last_year]["Month"].max()

    start_year = hist["Year"].min()
    last_index = (last_year - start_year) * 12 + last_month

    fut_rows = []
    for i in range(1, months_ahead + 1):
        m = last_month + i
        y = last_year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        q = ((m - 1) // 3) + 1
        mi = last_index + i
        fut_rows.append({
            "Year": int(y),
            "Month": int(m),
            "Quarter": int(q),
            "IsHolidaySL": 0,      
            "MonthIndex": int(mi),
        })
    future_df = pd.DataFrame(fut_rows)

    historical_avg = hist.groupby("Month").agg({
        "TotalOrders":"mean",
        "AvgUnitPrice":"mean",
        "AvgDiscount":"mean",
        "UniqueCustomers":"mean",
        "AvgShippingTime":"mean"
    }).reset_index()

    for col in ["TotalOrders", "AvgUnitPrice", "AvgDiscount", "UniqueCustomers", "AvgShippingTime"]:
        future_df[col] = future_df["Month"].apply(lambda m: historical_avg.loc[historical_avg["Month"]==m, col].values[0])

    X_future = future_df[feature_cols]
    X_future_scaled = scaler.transform(X_future)
    preds = model.predict(X_future_scaled)

    results = [{
        "Year": int(row["Year"]),
        "Month": int(row["Month"]),
        "Quarter": int(row["Quarter"]),
        "PredictedDemand": round(float(pred), 2)
    } for row, pred in zip(future_df.to_dict(orient="records"), preds)]

    return jsonify(results)

if __name__ == "__main__":
  app.run(debug=True)
