# from lib.SSH_pytorch.face_detection import ssh_detect
from lib.SSD.face_detection import ssd_detect
import cv2

if __name__ == '__main__':
    # img = cv2.imread('lib/SSH_pytorch/15.jpg')
    # print(img.shape)
    # boxes = detect(im = img)
    # print(boxes)

    path = 'lib/SSH_pytorch/15.jpg'
    pred = ssd_detect(path, target_shape=(360, 360), classify=True)
    print(pred)