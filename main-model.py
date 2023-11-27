# Connect to your personal Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Define Root Directory to Datasets
ROOT_DIR = '/content/gdrive/My Drive/Eco Report'

# Setting Environment
!pip install ultralytics

# Unzip Datasets (One time only)
!mkdir '/content/gdrive/My Drive/Eco Report/Datasets'
!unzip "/content/gdrive/MyDrive/Eco Report/aggra.v2i.yolov8.zip" -d "/content/gdrive/MyDrive/Eco Report/Datasets"

# Import Dataset Model
import os
from ultralytics import YOLO
from IPython.display import display, Image

# Load a model
model = YOLO("yolov8n.yaml")

# Callbacks
!mkdir '/content/gdrive/My Drive/Eco Report/logs'
data_config= os.path.join(ROOT_DIR, "trash-config.yaml")
log_dir= '/content/gdrive/My Drive/Eco Report/logs'
tensorboard_callback = model.train(data= data_config,
                                   epochs= 50,
                                   project= log_dir,
                                   )

# Validate Model
!yolo task=detect mode=val model='/content/gdrive/My Drive/Eco Report/logs/detect/train/weights/best.pt' data=trash-config.yaml

# Result Check
Image(filename=f'(/content/gdrive/My Drive/Eco Report/logs/train/result.png)', width=600)
Image(filename=f'(/content/gdrive/My Drive/Eco Report/logs/train/confusion_matrix.png)', width=600)

# Predict
!yolo task=detect mode=predict model='/content/gdrive/My Drive/Eco Report/logs/detect/train/weights/best.pt' conf=0.25 source=test_images
!yolo task=segment mode=predict model='/content/gdrive/My Drive/Eco Report/logs/detect/train/weights/best.seg' source=test_images
!yolo task=detect mode=predict model='/content/gdrive/My Drive/Eco Report/logs/detect/train/weights/best.pt'source=test_images
