from multiprocessing import Process
from sales_prediction_model import app as sales_app
from sales_prediction_model_rf import app as sales_app_rf
from regional_sales_model import app as region_app
from demand_model import app as product_app
from customer_model import app as customer_app

PORTS = {
    'sales': 5000,
    'region': 5001,
    'product': 5002,
    'customer': 5003,
    'sales_rf': 5004
}

# Functions to run each app
def run_sales():
    sales_app.run(port=PORTS['sales'], debug=True, use_reloader=False)

def run_sales_rf():
    sales_app_rf.run(port=PORTS['sales_rf'], debug=True, use_reloader=False)

def run_region():
    region_app.run(port=PORTS['region'], debug=True, use_reloader=False)

def run_product():
    product_app.run(port=PORTS['product'], debug=True, use_reloader=False)

def run_customer():
    customer_app.run(port=PORTS['customer'], debug=True, use_reloader=False)

if __name__ == "__main__":
    processes = [
        Process(target=run_sales),
        Process(target=run_region),
        Process(target=run_product),
        Process(target=run_customer),
        Process(target=run_sales_rf)
    ]

    # Start all processes
    for p in processes:
        p.start()

    print("All model Flask apps are running.")

    # Wait for processes to finish
    for p in processes:
        p.join()