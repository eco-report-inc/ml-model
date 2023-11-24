# Connect to your personal Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Define Root Directory to Datasets
ROOT_DIR = '/content/gdrive/My Drive/...' # Fill in the blank with the path of your Datasets Directories

# Setting Environment
!pip install ultralytics

# Unzip Your Datasets (Do it one time only, if you got disconnected from Google Collab, run all the cell except this one)
1unzip "/content/gdrive/MyDrive/...(your dataset path)" -d "/content/gdrive/MyDrive/...(your desired path for unzip datasets)"

# Import Dataset Model
import os
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")

# Use the model
model.train(data= os.path.join(ROOT_DIR, "trash-config.yaml"), epochs=3) # train the model and custom the epoch to reach your desired accuracy

# Result Check
from IPython.display import display, Image
Image(filename=f'(fill it with your result.png path)', width=600)

Image(filename=f'(fill it with your confusion_matrix.png path)', width=600)
