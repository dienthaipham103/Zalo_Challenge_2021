from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import shutil
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='data/train_faces',
	help="path to input dataset")
ap.add_argument("-r", "--bin", default='data/train_faces/wrong_label',
	help="folder that contains wrong labels")
args = vars(ap.parse_args())

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
modelPath = 'model/mobilenet_classifier_1.model'
model = load_model(modelPath)


folder = args['dataset']
wrong_folder = args['bin']

def remove(label):
    imgFolder = os.path.join(folder, label)
    wrongImgFolder = os.path.join(wrong_folder, label)

    for filename in os.listdir(imgFolder):
        path = os.path.join(imgFolder, filename)
        face = cv2.imread(path)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # keras.image: rgb, cv2: bgr

        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # pass the face through the model to determine if the face
        # has a mask or not
        (mask, withoutMask) = model.predict(face)[0]

        # determine the class label and color we'll use to draw
        # the bounding box and text
        class_id = 0 if mask > withoutMask else 1
        conf = max(mask, withoutMask)

        if (class_id == 0 and label == 'no_mask') or (class_id == 1 and label=='mask'):
            print('{}: {}'.format(filename, conf))
            saved_path = os.path.join(wrongImgFolder, filename)
            shutil.move(path, saved_path)

remove('mask')
remove('no_mask')