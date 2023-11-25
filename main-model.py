# Connect to your personal Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Define Root Directory to Datasets
ROOT_DIR = '/content/gdrive/My Drive/Eco Report'

# Setting Environment
!pip install ultralytics

# Unzip Datasets (One time only)
!mkdir '/content/gdrive/My Drive/Eco Report/Datasets'
!unzip "/content/gdrive/MyDrive/Eco Report/aggra.v2i.yolov8.zip)" -d "/content/gdrive/MyDrive/Eco Report/Datasets)"

# Import Dataset Model
import os
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")

# Callbacks
!mkdir '/content/gdrive/My Drive/Eco Report/logs'
data_config= os.path.join(ROOT_DIR, "trash-config.yaml")
log_dir= '/content/gdrive/My Drive/Eco Report/logs'
tensorboard_callback = model.train(data= data_config,
                                   epochs= 100,
                                   project= log_dir,
                                   )
# Use the model
model.train()

# Result Check
from IPython.display import display, Image
Image(filename=f'(/content/gdrive/My Drive/Eco Report/logs/train/result.png)', width=600)
Image(filename=f'(/content/gdrive/My Drive/Eco Report/logs/train/confusion_matrix.png)', width=600)
