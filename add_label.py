from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import shutil

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model("model/mobilenet_classifier.model")

# mask folder
# folder = 'large_faces/mask'
# saved_folder = 'large_faces/mask_removed'

'''
Use pre-trained model to get labels of unlabeled faces. Add more faces without mask to the dataset to avoid imbalance
'''
def label(folder):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
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

        if conf > 0.7 and class_id == 1:
            saved_folder = os.path.join('data/train_faces/no_mask', filename)
            shutil.move(path, saved_folder)

# relabel('large_faces/mask_removed')
# relabel('large_faces/no_mask_removed')
label('data/train_faces/nan')