from flask import Flask, request, jsonify
import pyodbc
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = Flask(__name__)
MODEL_FILE = "sales_prediction_model.pkl"
SCALER_FILE = "scaler.pkl"
X_COLUMNS_FILE = "X_columns.pkl"

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
    SELECT LineTotal,
           OrderQty, 
           UnitPrice, 
           UnitPriceDiscount, 
           ProductKey, 
           CustomerKey, 
           TerritoryKey, 
           SalesPersonKey, 
           OrderDateKey
    FROM FactSalesOrderDetail
    """
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Train model endpoint
@app.route('/train_sales', methods=['POST', 'GET'])
def train_model():
    df = fetch_data()
    
    y = df['LineTotal']
    numeric_features = ['OrderQty', 'UnitPrice', 'UnitPriceDiscount']
    X_numeric = df[numeric_features]
    
    categorical_features = ['ProductKey', 'CustomerKey', 'TerritoryKey', 'SalesPersonKey']
    X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
    
    df['OrderDateKey'] = pd.to_datetime(df['OrderDateKey'])
    df['OrderYear'] = df['OrderDateKey'].dt.year
    df['OrderMonth'] = df['OrderDateKey'].dt.month
    df['OrderDay'] = df['OrderDateKey'].dt.day
    X_time = df[['OrderYear', 'OrderMonth', 'OrderDay']]
    
    X = pd.concat([X_numeric, X_categorical, X_time], axis=1).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Save model, scaler, and columns
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(X.columns, X_COLUMNS_FILE)
    
    return jsonify({"message": "Linear Regression model trained successfully"})

# Prediction endpoint
@app.route('/predict_sales', methods=['GET'])
def predict_sales():
    months_ahead = int(request.args.get("months", 6))  # default 6 months
    year = request.args.get("year", None)
    month = request.args.get("month", None)
    
    # Load model, scaler, columns
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    X_columns = joblib.load(X_COLUMNS_FILE)
    
    df_train = fetch_data()
    numeric_features = ['OrderQty', 'UnitPrice', 'UnitPriceDiscount']
    categorical_features = ['ProductKey', 'CustomerKey', 'TerritoryKey', 'SalesPersonKey']
    
    avg_numeric = df_train[numeric_features].mean().to_dict()
    top_cats = {col: df_train[col].mode()[0] for col in categorical_features}
    
    # Determine starting point
    df_train['OrderDateKey'] = pd.to_datetime(df_train['OrderDateKey'])
    last_year = df_train['OrderDateKey'].dt.year.max()
    last_month = df_train['OrderDateKey'].dt.month.max()
    
    # If year/month provided, use that as starting point
    if year and month:
        start_year = int(year)
        start_month = int(month)
    else:
        start_year = last_year
        start_month = last_month
    
    # Generate future months
    future_dates = []
    for i in range(1, months_ahead + 1):
        month_num = start_month + i
        year_num = start_year + (month_num - 1) // 12
        month_num = ((month_num - 1) % 12) + 1
        future_dates.append((year_num, month_num))
    
    # Create DataFrame for prediction
    X_future_rows = []
    for y_val, m_val in future_dates:
        X_numeric_df = pd.DataFrame([avg_numeric])
        X_categorical_df = pd.DataFrame({f"{col}_{val}": [1] for col, val in top_cats.items()})
        X_time_df = pd.DataFrame({'OrderYear': [y_val], 'OrderMonth': [m_val], 'OrderDay': [1]})
        X_row = pd.concat([X_numeric_df, X_categorical_df, X_time_df], axis=1).fillna(0)
        # Align columns
        for col in X_columns:
            if col not in X_row.columns:
                X_row[col] = 0
        X_row = X_row[X_columns]
        X_future_rows.append(X_row)
    
    X_future = pd.concat(X_future_rows, ignore_index=True)
    X_scaled = scaler.transform(X_future)
    preds = model.predict(X_scaled)
    
    results = [
        {"Year": int(y_val), "Month": int(m_val), "PredictedSales": float(round(preds[i], 2))}
        for i, (y_val, m_val) in enumerate(future_dates)
    ]
    
    return jsonify(results)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
