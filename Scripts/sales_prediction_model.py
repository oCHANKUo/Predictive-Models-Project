# Prediction Target = line total
# Features = OrderQty, UnitPrice, UnitPriceDiscount, ProductKey, CustomerKey, TerritoryKey, SalesPersonKey, OrderDateKey

import pyodbc 
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import StandardScaler

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


#Funtion to train the model
train = df[df['OrderDateKey'] < '2012-01-01']
test = df[df['OrderDateKey'] >= '2012-01-01']

X_train = X.loc[train.index]
X_test = X.loc[test.index]
y_train = y.loc[train.index]
y_test = y.loc[test.index]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)


# Prediction

print(X_time.head())