from threading import Thread

# Import the apps from your model files
from sales_prediction_model import app as sales_app
from regional_sales_model import app as region_app
from demand_model import app as product_app
from customer_model import app as customer_app

# Define the ports for each app
PORTS = {
    'sales': 5000,
    'region': 5001,
    'product': 5002,
    'customer': 5003
}

# Functions to run each app
def run_sales():
    sales_app.run(port=PORTS['sales'], debug=True, use_reloader=False)

def run_region():
    region_app.run(port=PORTS['region'], debug=True, use_reloader=False)

def run_product():
    product_app.run(port=PORTS['product'], debug=True, use_reloader=False)

def run_customer():
    customer_app.run(port=PORTS['customer'], debug=True, use_reloader=False)

# Start all apps in separate threads
if __name__ == "__main__":
    Thread(target=run_sales).start()
    Thread(target=run_region).start()
    Thread(target=run_product).start()
    Thread(target=run_customer).start()
    print("All model Flask apps are running.")
