import cv2
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

threshold = 32

def isSmall(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if h <= threshold or w <= threshold:
        return True

def removeSmall(folder, saved_folder='data/train_faces/small_faces'):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if isSmall(path):
            saved_path = os.path.join(saved_folder, filename)
            shutil.move(path, saved_path)

def resizeImages(folder):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        image = load_img(path, target_size=(224, 224))
        image.save(path)

if __name__ == '__main__':
    # Remove small faces
    print('Removing small faces...')
    removeSmall(folder='data/train_faces/mask')
    removeSmall(folder='data/train_faces/no_mask')
    removeSmall(folder='data/train_faces/nan')

    # print('Resize images ...')
    # resizeImages('data/train_faces/mask')

    '''
    add label
    train
    remove wrong label
    train
    ---> final
    '''