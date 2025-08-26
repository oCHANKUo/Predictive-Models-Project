from flask import Flask, request, jsonify
from flask_cors import CORS
import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
import os

app = Flask(__name__)
CORS(app)

# ---------------- Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
SCALER_DIR = os.path.join(ROOT_DIR, "scaler")
COLUMNS_DIR = os.path.join(ROOT_DIR, "columns")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(COLUMNS_DIR, exist_ok=True)

CLASSIFIER_FILE = os.path.join(MODELS_DIR, "region_sales_model.pkl")
SALES_MODEL_FILE = os.path.join(MODELS_DIR, "region_sales_sales_model.pkl")
SCALER_FILE = os.path.join(SCALER_DIR, "region_sales_scaler.pkl")
COLUMN_FILE = os.path.join(COLUMNS_DIR, "region_sales_columns.pkl")

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER03;DATABASE=DataWarehouseClassic;UID=admin;PWD=admin"
    )
    return conn

def fetch_data():
    query = """
        SELECT 
            t.TerritoryName,
            p.CategoryName,
            d.Year,
            d.Month,
            SUM(f.OrderQty) AS TotalSales
        FROM FactSalesOrderDetail f
        JOIN DimProduct p ON f.ProductKey = p.ProductKey
        JOIN DimTerritory t ON f.TerritoryKey = t.TerritoryKey
        JOIN DimDate d ON f.OrderDateKey = d.DateKey
        GROUP BY t.TerritoryName, p.CategoryName, d.Year, d.Month
        ORDER BY d.Year, d.Month
    """
    conn = get_connection()
    df = pd.read_sql(query, conn)
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    conn.close()
    return df

def preprocess(df):
    pivot = df.pivot_table(
        index=['TerritoryName', 'Year', 'Month'],
        columns='CategoryName',
        values='TotalSales',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    pivot['TopCategory'] = pivot.drop(columns=['TerritoryName','Year','Month']).idxmax(axis=1)

    X_cls = pd.get_dummies(pivot['TerritoryName'], drop_first=True)
    y_cls = pivot['TopCategory']

    # MonthIndex for trend in regression
    pivot['MonthIndex'] = (pivot['Year'].astype(int) - pivot['Year'].min())*12 + pivot['Month'].astype(int)
    X_reg = X_cls.copy()
    X_reg['MonthIndex'] = pivot['MonthIndex']
    y_reg = pivot.drop(columns=['TerritoryName','Year','Month','TopCategory']).sum(axis=1)

    return X_cls, y_cls, X_reg, y_reg, pivot

@app.route("/train_regional_sales", methods=['POST','GET'])
def train_model():
    df = fetch_data()
    X_cls, y_cls, X_reg, y_reg, pivot = preprocess(df)

    # Classifier
    x_train, x_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save classifier
    with open(CLASSIFIER_FILE, "wb") as f:
        pickle.dump(clf, f)

    # Regressor
    # Optional: scale MonthIndex
    scaler = StandardScaler()
    X_reg_scaled = X_reg.copy()
    X_reg_scaled[['MonthIndex']] = scaler.fit_transform(X_reg[['MonthIndex']])
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_reg_scaled, y_reg)

    # Save regressor, scaler, and columns
    with open(SALES_MODEL_FILE, "wb") as f:
        pickle.dump(reg, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    with open(COLUMN_FILE, "wb") as f:
        pickle.dump(X_reg.columns.tolist(), f)

    return jsonify({"message":"Regional Sales Model Trained Successfully"})

@app.route("/predict_regional_sales", methods=['GET','POST'])
def predict():
    months_to_predict = int(request.args.get("months", 6))

    if not os.path.exists(CLASSIFIER_FILE) or not os.path.exists(SALES_MODEL_FILE):
        return jsonify({"error": "Model not trained yet"}), 400

    # Load models
    with open(CLASSIFIER_FILE, "rb") as f:
        clf = pickle.load(f)
    with open(SALES_MODEL_FILE, "rb") as f:
        reg = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(COLUMN_FILE, "rb") as f:
        columns = pickle.load(f)

    df = fetch_data()
    X_cls, y_cls, X_reg, y_reg, pivot = preprocess(df)

    last_year = df['Year'].max()
    last_month = df[df['Year']==last_year]['Month'].max()

    # Determine territories to predict
    territory_input = request.form.get("TerritoryName") or request.args.get("TerritoryName")
    data = request.get_json(silent=True)
    territory_input = territory_input or (data.get("TerritoryName") if data else None)
    territories = [territory_input] if territory_input else pivot['TerritoryName'].unique()

    results = []
    for terr in territories:
        for i in range(1, months_to_predict + 1):
            # Compute month and year
            month = last_month + i
            year = last_year + (month - 1) // 12
            month = ((month - 1) % 12) + 1

            # ---------------- Class prediction ----------------
            row_cls = pd.DataFrame([terr], columns=['TerritoryName'])
            row_cls = pd.get_dummies(row_cls, drop_first=True)
            row_cls = row_cls.reindex(columns=X_cls.columns, fill_value=0)
            top_category = clf.predict(row_cls)[0]

            row_reg = row_cls.copy()

            month_index = (year - df['Year'].min()) * 12 + month
            row_reg['MonthIndex'] = month_index

            # Scale MonthIndex
            row_reg_scaled = row_reg.copy()
            row_reg_scaled['MonthIndex'] = scaler.transform(row_reg[['MonthIndex']])

            # Align columns for regressor
            row_reg_scaled = row_reg_scaled.reindex(columns=columns, fill_value=0)

            predicted_sales = reg.predict(row_reg_scaled)[0]

            results.append({
                "TerritoryName": str(terr),
                "Year": int(year),
                "Month": int(month),
                "PredictedTopCategory": str(top_category),
                "PredictedSales": float(predicted_sales)
            })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
