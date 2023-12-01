# Connect to your personal Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Define Root Directory to Datasets
ROOT_DIR = '/content/gdrive/My Drive/Eco Report'

# Setting Environment
!pip install ultralytics

# Unzip Datasets (One time only)
!unzip "/content/gdrive/MyDrive/Eco Report/aggra.v2i.yolov8.zip" -d "/content/gdrive/MyDrive/Eco Report/Datasets"

# Import Library
import os
import shutil
from ultralytics import YOLO
from IPython.display import display, Image

# Load a model
model = YOLO("yolov8n.yaml")

# Training Logs
data_config= os.path.join(ROOT_DIR, "trash-config.yaml")
log_dir= '/content/gdrive/My Drive/Eco Report/logs'

# Train a model
model.train(data= data_config,
            epochs= 20,
            project= log_dir,
            )

# Train Result Check
directory_path = '/content/gdrive/MyDrive/Eco Report/logs/train'
contents = os.listdir(directory_path)
print(contents)

Image(filename=f'/content/gdrive/My Drive/Eco Report/logs/train/result.png', width=600)
Image(filename=f'/content/gdrive/My Drive/Eco Report/logs/train/confusion_matrix.png', width=600)

# Validate Model
output_directory = '/content/gdrive/MyDrive/EcoReport'
!yolo task=detect mode=val model='/content/gdrive/MyDrive/EcoReport/logs/train8/weights/best.pt' data= '/content/gdrive/MyDrive/EcoReport/data.yaml'

shutil.move('/content/runs/detect/val', output_directory)

# Check Validation Result
Image(filename=f'/content/gdrive/MyDrive/EcoReport/val/confusion_matrix.png', width=600)

# Predict Model
model= YOLO('/content/gdrive/MyDrive/Eco Report/logs/train4/weights/best.pt')
source= '/content/gdrive/MyDrive/Eco Report/Datasets/test/images/-libre-place-des-dejections-en-plusieurs-langues-photo-marc-wirtz-1430300655_jpg.rf.0d5eae0d8975922bffeafd619e49f2f5.jpg'
result= model(source)

model.predict(source, save=True, imgsz=640, conf=0.25)

# Check Prediction Result
Image(filename=f'/content/runs/detect/predict2/-libre-place-des-dejections-en-plusieurs-langues-photo-marc-wirtz-1430300655_jpg.rf.0d5eae0d8975922bffeafd619e49f2f5.jpg', width=600)

