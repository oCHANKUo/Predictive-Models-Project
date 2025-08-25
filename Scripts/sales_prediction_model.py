# Prediction Target = line total
# Features = OrderQty, UnitPrice, UnitPriceDiscount, ProductKey, CustomerKey, TerritoryKey, SalesPersonKey, OrderDateKey

import pyodbc 
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

model = linear_model.LinearRegression()
scaler = StandardScaler()

# First import data
# Function to connect to the database
def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER03;"
        "DATABASE=DataWarehouseClassic;"
        "UID=admin;"
        "PWD=admin"
    )

    return conn

# Function to import the data, it will connect using the previous function
def get_data():
    conn = get_connection()
    query = """
    SELECT 
        LineTotal,
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
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# import data
df = get_data()

target = 'LineTotal'
y = df[target]

numeric_features = [
    'OrderQty', 
    'UnitPrice', 
    'UnitPriceDiscount'
]
X_numeric = df[numeric_features]

categorical_features = ['ProductKey', 
                        'CustomerKey', 
                        'TerritoryKey', 
                        'SalesPersonKey']
X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)

df['OrderDateKey'] = pd.to_datetime(df['OrderDateKey'])
# Extract year, month, day, day of week, quarter
df['OrderYear'] = df['OrderDateKey'].dt.year
df['OrderMonth'] = df['OrderDateKey'].dt.month
df['OrderDay'] = df['OrderDateKey'].dt.day
time_features = ['OrderYear', 'OrderMonth', 'OrderDay']
X_time = df[time_features]

X = pd.concat([X_numeric, X_categorical, X_time], axis=1)
X = X.fillna(0)

#Funtion to train the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)

joblib.dump(model, "sales_prediction_model.pkl")
joblib.dump(scaler, "scaler.pkl")


# Prediction
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}, R2: {r2}")