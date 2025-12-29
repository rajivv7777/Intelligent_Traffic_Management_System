import yaml

with open(r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicles\data.yaml", "r") as f:
    data = yaml.safe_load(f)
    print(data)
