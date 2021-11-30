import os
import cv2
import time
import numpy as np
from lib.SSH_pytorch.face_detection import ssh_detect
from lib.SSD.face_detection import ssd_detect
from func import im_show, SSH_init
from resnet_classifier import resnet_classify
from mobilenet_classifier import mobilenet_classify
from enhance_detection import merge_boxes


'''
Using SDD for detection (Including classification in detection problem)
'''
def draw(img_path, bboxs, img=None):
    if img is None:
        img = cv2.imread(img_path)
    img_cp = img.copy()

    if len(bboxs) == 0: return img

    for bbox in bboxs:
        class_id, conf, xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)


        prob_font_scale = ((xmax-xmin) * (ymax-ymin)) / (100 * 100)
        prob_font_scale = max(prob_font_scale, 0.25)
        prob_font_scale = min(prob_font_scale, 0.75)

        conf = max(0.0, conf)
        cv2.putText(img_cp, '{0:.2f}'.format(conf), (xmin + 7, ymin - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, prob_font_scale, (0, 0, 255), 1, lineType=cv2.LINE_AA)


        if class_id == 1:
            color = (0, 211, 255)
        elif class_id == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), color, 1)

    return img_cp

def predict_image(path, saved_path):
    ssd_bboxs = ssd_detect(im_path=path, target_shape=(360, 360), classify=True)
    ssh_bboxs = ssh_detect(path, classify=True)
    bboxs = merge_boxes(ssd_bboxs, ssh_bboxs, classify=True)

    img = draw(path, bboxs)
    # img = draw(path, ssd_bboxs)
    cv2.imwrite(saved_path, img)

def predict_folder(input_f='data/public_test', output_f='data/prediction'):
    for filename in os.listdir(input_f):
        print(filename)
        path = os.path.join(input_f, filename)
        saved_path = os.path.join(output_f, filename[:-4] + '.png')
        predict_image(path, saved_path)


if __name__ == "__main__":

    # demo_video(net, visualize=True)
    
    # uncomment below to run demo on video
    # demo_video(net, './data/videos/demo3.MOV', save_out=True, visualize=True)
    # path = 'test.jpg'
    # saved_path = 'result.png'
    # predict_image(path, saved_path)
    predict_folder(output_f='data/first_prediction')
