#STEP 2
import zipfile
import os
zip_path=r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\Army_vehicle.v1i.yolov8.zip"
zip_path=r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\Ambulance.v1i.yolov8.zip"
zip_path=r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\Firebrigade.v1i.yolov8.zip"
zip_path=r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\Police_car.v1i.yolov8.zip"
extract_to= r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicle"
os.makedirs(extract_to,exist_ok=True)
with zipfile.ZipFile(zip_path,'r')as zip_ref:
    zip_ref.extractall(extract_to)
    print("Data unzipped to :", extract_to)
