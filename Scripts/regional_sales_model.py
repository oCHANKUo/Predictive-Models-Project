from flask import Flask, request, jsonify
from flask_cors import CORS
import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

app = Flask(__name__)
CORS(app)

# ---------------- Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
COLUMNS_DIR = os.path.join(ROOT_DIR, "columns")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COLUMNS_DIR, exist_ok=True)

SALES_MODEL_FILE = os.path.join(MODELS_DIR, "regional_sales_model.pkl")
COLUMN_FILE = os.path.join(COLUMNS_DIR, "regional_sales_columns.pkl")

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
            d.Year,
            d.Month,
            SUM(f.OrderQty) AS TotalSales
        FROM FactSalesOrderDetail f
        JOIN DimTerritory t ON f.TerritoryKey = t.TerritoryKey
        JOIN DimDate d ON f.OrderDateKey = d.DateKey
        GROUP BY t.TerritoryName, d.Year, d.Month
        ORDER BY d.Year, d.Month
    """
    conn = get_connection()
    df = pd.read_sql(query, conn)
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    conn.close()
    return df

def preprocess(df):
    # Add MonthIndex for trend
    df['MonthIndex'] = (df['Year'] - df['Year'].min())*12 + df['Month']

    # Features = Territory one-hot + MonthIndex
    X = pd.get_dummies(df[['TerritoryName']], drop_first=True)
    X['MonthIndex'] = df['MonthIndex']

    y = df['TotalSales']

    return X, y, df

@app.route("/train_regional_sales", methods=['POST','GET'])
def train_model():
    df = fetch_data()
    X, y, df_proc = preprocess(df)

    # Train regression model
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X, y)

    with open(SALES_MODEL_FILE, "wb") as f:
        pickle.dump(reg, f)
    with open(COLUMN_FILE, "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    return jsonify({"message":"Regional Sales Model Trained Successfully"})

@app.route("/predict_regional_sales", methods=['GET','POST'])
def predict():
    months_to_predict = int(request.args.get("months", 6))

    if not os.path.exists(SALES_MODEL_FILE):
        return jsonify({"error": "Model not trained yet"}), 400

    # Load model + columns
    with open(SALES_MODEL_FILE, "rb") as f:
        reg = pickle.load(f)
    with open(COLUMN_FILE, "rb") as f:
        columns = pickle.load(f)

    df = fetch_data()
    X, y, df_proc = preprocess(df)
    last_year = df['Year'].max()
    last_month = df[df['Year']==last_year]['Month'].max()

    # Determine territories
    territory_input = request.form.get("TerritoryName") or request.args.get("TerritoryName")
    data = request.get_json(silent=True)
    territory_input = territory_input or (data.get("TerritoryName") if data else None)
    territories = [territory_input] if territory_input else df['TerritoryName'].unique()

    results = []
    for terr in territories:
        for i in range(1, months_to_predict+1):
            month = last_month + i
            year = last_year + (month-1)//12
            month = ((month-1)%12) + 1

            # Build row
            row = pd.DataFrame(0, index=[0], columns=columns)
            # Territory encoding
            terr_col = [c for c in columns if terr in c]
            if terr_col:
                row.loc[0, terr_col] = 1
            # Month index
            month_index = int((year - df['Year'].min())*12 + month)
            row.loc[0, 'MonthIndex'] = month_index

            predicted_sales = reg.predict(row)[0]

            results.append({
                "TerritoryName": str(terr),
                "Year": int(year),
                "Month": int(month),
                "PredictedSales": float(predicted_sales)
            })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
