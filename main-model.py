# Connect to your personal Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Define Root Directory to Datasets
ROOT_DIR = '/content/gdrive/My Drive/Eco Report'

# Setting Environment
!pip install ultralytics

# Unzip Datasets (One time only)
!unzip "/content/gdrive/MyDrive/Eco Report/aggra.v2i.yolov8.zip" -d "/content/gdrive/MyDrive/Eco Report/Datasets"

# Import Dataset Model
import os
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

# Result Check
directory_path = '/content/gdrive/MyDrive/Eco Report/logs/train'
contents = os.listdir(directory_path)
print(contents)

Image(filename=f'(/content/gdrive/My Drive/Eco Report/logs/train/result.png)', width=600)
Image(filename=f'(/content/gdrive/My Drive/Eco Report/logs/train/confusion_matrix.png)', width=600)

# When predicting model if you got UTF-8 error please run this few code first
import locale
print(locale.getpreferredencoding())

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# Predict Model
model= YOLO('/content/gdrive/MyDrive/Eco Report/logs/train4/weights/best.pt')
source= '/content/gdrive/MyDrive/Eco Report/Datasets/test/images/-libre-place-des-dejections-en-plusieurs-langues-photo-marc-wirtz-1430300655_jpg.rf.0d5eae0d8975922bffeafd619e49f2f5.jpg'
result= model(source)

model.predict(source, save=True, imgsz=640, conf=0.25)

# Check predicted images
Image(filename=f'/content/runs/detect/predict2/-libre-place-des-dejections-en-plusieurs-langues-photo-marc-wirtz-1430300655_jpg.rf.0d5eae0d8975922bffeafd619e49f2f5.jpg', width=600)

# Haven't try this code yet #
!yolo task=detect mode=predict model='/content/gdrive/My Drive/Eco Report/logs/detect/train/weights/best.pt' conf=0.25 source=test_images
!yolo task=segment mode=predict model='/content/gdrive/My Drive/Eco Report/logs/detect/train/weights/best.seg' source=test_images
!yolo task=detect mode=predict model='/content/gdrive/My Drive/Eco Report/logs/detect/train/weights/best.pt'source=test_images
