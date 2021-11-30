from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
# model = load_model("classification.model")
pretrained_model = load_model("model/mobilenet_classifier.model")

'''
Img input (rgb)
Return: No(1)/Mask(0) and prob(1)
'''
def mobilenet_classify(img_arr):

    face = cv2.resize(img_arr, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    (mask, withoutMask) = pretrained_model.predict(face)[0]

    # determine the class label and color we'll use to draw
    # the bounding box and text
    class_id = 0 if mask > withoutMask else 1
    conf = withoutMask

    return class_id, conf 