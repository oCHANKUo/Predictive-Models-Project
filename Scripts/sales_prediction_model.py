# Prediction Target = line total
# Features = OrderQty, UnitPrice, UnitPriceDiscount, ProductKey, CustomerKey, TerritoryKey, SalesPersonKey, OrderDateKey

import pyodbc 
from sklearn import linear_model
import pandas as pd

# First import data
def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost/MSSQLSERVER03;"
        "DATABASE=DataWarehouseClassic;"
        "UID=admin;"
        "PWD=admin"
    )
    return conn

def get_data():
    conn = get_connection()
    query = """
    SELECT 
        f.OrderQty, 
        f.UnitPrice, 
        f.UnitPriceDiscount, 
        f.ProductKey, 
        f.CustomerKey, 
        f.TerritoryKey, 
        f.SalesPersonKey, 
        f.OrderDateKey 
    FROM FactSales f
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df



model = linear_model.LinearRegression()