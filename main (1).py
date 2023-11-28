# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H-lx46NomGOO5oGDq4bYpzHf21uWsRaF

# Setting Environment
"""

from google.colab import drive

drive.mount('/content/gdrive')

ROOT_DIR = '/content/gdrive/My Drive/Eco Report'

!pip install ultralytics

"""# Import and Train Dataset Model"""

!mkdir '/content/gdrive/My Drive/Eco Report/logs'

import os
from ultralytics import YOLO
from tensorflow import keras

# Load a model
model = YOLO("yolov8n.yaml")

# Set up
data_config= os.path.join(ROOT_DIR, "trash-config.yaml")
logs_dir= '/content/gdrive/My Drive/Eco Report/logs'

# Train a model
model_train = model.train(data= data_config,
                                   epochs= 25,
                                   project= logs_dir,
                                   )

"""# Result Check"""

directory_path = '/content/gdrive/MyDrive/Eco Report/logs/train4'
contents = os.listdir(directory_path)
print(contents)

from IPython.display import display, Image

Image(filename=f'/content/gdrive/MyDrive/Eco Report/logs/train4/results.png', width=600)

Image(filename=f'/content/gdrive/MyDrive/Eco Report/logs/train4/confusion_matrix_normalized.png', width=600)

import locale
print(locale.getpreferredencoding())

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

model= YOLO('/content/gdrive/MyDrive/Eco Report/logs/train4/weights/best.pt')
source= '/content/gdrive/MyDrive/Eco Report/Datasets/test/images/-libre-place-des-dejections-en-plusieurs-langues-photo-marc-wirtz-1430300655_jpg.rf.0d5eae0d8975922bffeafd619e49f2f5.jpg'

result= model(source)

model.predict(source, save=True, imgsz=640, conf=0.25)