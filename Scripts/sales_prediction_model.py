# Prediction Target = line total
# Features = OrderQty, UnitPrice, LineTotal

import pyodbc 
from sklearn import linear_model



model = linear_model.LinearRegression()