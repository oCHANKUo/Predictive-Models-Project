from flask import Flask, request, jsonify
from flask_cors import CORS
import pyodbc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)
CORS(app)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
SCALER_DIR = os.path.join(ROOT_DIR, "scaler")
COLUMNS_DIR = os.path.join(ROOT_DIR, "columns")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(COLUMNS_DIR, exist_ok=True)

CUSTOMER_MODEL_FILE = os.path.join(MODELS_DIR, "customer_model.pkl")
CUSTOMER_SCALER_FILE = os.path.join(SCALER_DIR, "customer_scaler.pkl")
CUSTOMER_COLUMNS_FILE = os.path.join(COLUMNS_DIR, "customer_columns.pkl")

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER03;DATABASE=DataWarehouseClassic;UID=admin;PWD=admin"
    )
    return conn

def fetch_customer_data():
    query = """
    SELECT 
        c.CustomerKey,
        d.Year,
        d.Month,
        SUM(f.TotalDue) AS TotalSpent,
        SUM(f.OrderQty) AS TotalQuantity,
        COUNT(f.SalesOrderID) AS PurchaseCount,
        c.Gender,
        c.EmailPromotion,
        c.CountryRegionName
    FROM FactSalesOrderDetail f
    JOIN DimDate d ON f.OrderDateKey = d.DateKey
    JOIN DimCustomer c ON f.CustomerKey = c.CustomerKey
    GROUP BY c.CustomerKey, d.Year, d.Month, c.Gender, c.EmailPromotion, c.CountryRegionName
    ORDER BY c.CustomerKey, d.Year, d.Month;
    """
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def prepare_customer_data(df, fit_scaler=True, scaler=None, columns=None):
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df = df.sort_values(by=['CustomerKey', 'Year', 'Month'])

    df['NextPurchase'] = df.groupby('CustomerKey')['PurchaseCount'].shift(-1)
    df['NextPurchase'] = df['NextPurchase'].apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna(subset=['NextPurchase'])

    df_encoded = pd.get_dummies(df, columns=['Gender', 'CountryRegionName', 'EmailPromotion'], drop_first=True)

    X = df_encoded.drop(columns=['CustomerKey', 'NextPurchase'])
    y = df_encoded['NextPurchase'].astype(int)

    numeric_cols = ['TotalSpent', 'TotalQuantity', 'Year', 'Month']
    if fit_scaler:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    if columns:
        for col in columns:
            if col not in X.columns:
                X[col] = 0
        X = X[columns]
    else:
        columns = X.columns.tolist()

    return X, y, df_encoded, scaler, columns

@app.route('/train_customer', methods=['POST', 'GET'])
def train_customer_model():
    df = fetch_customer_data()
    X, y, _, scaler, columns = prepare_customer_data(df, fit_scaler=True)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(CUSTOMER_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(CUSTOMER_SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    with open(CUSTOMER_COLUMNS_FILE, "wb") as f:
        pickle.dump(columns, f)

    return jsonify({"message": "Customer purchase behavior model trained successfully"})

@app.route('/predict_customer', methods=['GET', 'POST'])
def predict_customer():
    if not os.path.exists(CUSTOMER_MODEL_FILE):
        return jsonify({"error": "Model not trained. Call /train_customer first"}), 400

    # Load model, scaler, and columns
    with open(CUSTOMER_MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(CUSTOMER_SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(CUSTOMER_COLUMNS_FILE, "rb") as f:
        columns = pickle.load(f)

    # Prepare customer data
    df = fetch_customer_data()
    X, _, full_df, _, _ = prepare_customer_data(df, fit_scaler=False, scaler=scaler, columns=columns)

    # Predict
    preds = model.predict_proba(X)[:, 1]
    full_df['PurchaseProbability'] = preds
    full_df['Prediction'] = (preds > 0.5).astype(int)

    data = request.get_json(silent=True) or {}
    selected_year = request.args.get("year", type=int) 
    selected_month = request.args.get("month", type=int)
    top_n = request.args.get("top_n", default=10, type=int)

    latest = full_df.copy()
    if selected_year:
        latest = latest[latest['Year'] == selected_year]
    if selected_month:
        latest = latest[latest['Month'] == selected_month]

    # Keep only the latest entry per customer
    latest = latest.groupby("CustomerKey").tail(1)

    top_n = request.args.get("top_n", default=10, type=int)
    latest = latest.sort_values(by="PurchaseProbability", ascending=False).head(top_n)

    results = latest[['CustomerKey', 'Year', 'Month', 'PurchaseProbability', 'Prediction']].to_dict(orient="records")
    return jsonify(results)

# if __name__ == '__main__':
#    app.run(debug=True)