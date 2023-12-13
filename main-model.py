# Connect to your personal Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Define Root Directory to Datasets
ROOT_DIR = '/content/gdrive/My Drive/EcoReport'

# Setting Environment
!pip install ultralytics

# Unzip Datasets (One time only)
!unzip "/content/gdrive/MyDrive/EcoReport/train5-20231213T045433Z-001.zip" -d "/content/gdrive/MyDrive/EcoReport/logs"

# Import Library
import os
import shutil
from ultralytics import YOLO
from IPython.display import display, Image

# Base Model & Training Logs
model = YOLO("yolov8n.yaml")
data_config= os.path.join(ROOT_DIR, "trash-config.yaml")
log_dir= '/content/gdrive/My Drive/Eco Report/logs'

# Train and Validate model
model.train(data= data_config,
            pretrained= False,
            epochs= 300,
            batch= 16,
            imgsz= 640,
            optimizer= 'Adam',
            lr0=0.001,
            weight_decay=0.0005,
            patience= 50,
            val= True,
            project= logs_dir,
            )

# Train Result Check
directory_path = '/content/gdrive/MyDrive/Eco Report/logs/train'
contents = os.listdir(directory_path)
print(contents)

Image(filename=f'/content/gdrive/My Drive/Eco Report/logs/train/results.png', width=600)
Image(filename=f'/content/gdrive/My Drive/Eco Report/logs/train/confusion_matrix.png', width=600)

# Predict Model in Local
model= YOLO('/content/gdrive/MyDrive/Eco Report/logs/train/weights/best.pt')
source= '/content/gdrive/MyDrive/Eco Report/Datasets/test/images/-libre-place-des-dejections-en-plusieurs-langues-photo-marc-wirtz-1430300655_jpg.rf.0d5eae0d8975922bffeafd619e49f2f5.jpg'
result= model.predict(source, save=True, imgsz=640, conf=0.25)

# Check Prediction Result
Image(filename=f'/content/runs/detect/predict2/-libre-place-des-dejections-en-plusieurs-langues-photo-marc-wirtz-1430300655_jpg.rf.0d5eae0d8975922bffeafd619e49f2f5.jpg', width=600)

# Save Predicted Image
output_directory = '/content/gdrive/MyDrive/EcoReport/Result'
shutil.move('/content/runs/detect/predict', output_directory)

#Export Model format ONNX
model= YOLO('/content/gdrive/MyDrive/EcoReport/logs/train/weights/best.pt')
model.export(format='tflite')
