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


def draw(img_path, bboxs, img=None, thresh=0.5, max_size=0):
    if img is None:
        img = cv2.imread(img_path)
    img_cp = img.copy()

    len_line = int(img_cp.shape[1] / 5)
    pad_percent = int(img_cp.shape[1] / 2)
    x = int(img_cp.shape[1] / 25)
    y = int(img_cp.shape[0] / 25)
    pad_x = int(img_cp.shape[1] / 50)
    pad_y = int(img_cp.shape[0] / 25)
    pad_text = 5
    font_scale = (img_cp.shape[0] * img_cp.shape[1]) / (750 * 750)
    font_scale = max(font_scale, 0.25)
    font_scale = min(font_scale, 0.75)

    font_thickness = 1
    if max(img_cp.shape[0], img_cp.shape[1]) > 750: font_thickness = 2

    if bboxs.shape[0] == 0: return img
    bboxs = bboxs[np.where(bboxs[:, -1] > thresh)[0]]
    bboxs = bboxs.astype(int)

    cnt_mask = 0
    cnt_nomask = 0

    for bbox in bboxs:
        img_bbox = img[bbox[1]:bbox[3], bbox[0]:bbox[2], [2, 1, 0]]

        if img_bbox.shape[0] * img_bbox.shape[1] < max_size:
            continue

        (type, prob) = mobilenet_classify(img_arr=img_bbox)
        prob_font_scale = (img_bbox.shape[0] * img_bbox.shape[1]) / (100 * 100)
        prob_font_scale = max(prob_font_scale, 0.25)
        prob_font_scale = min(prob_font_scale, 0.75)

        cv2.putText(img_cp, '{0:.2f}'.format(1-prob), (bbox[0] + 7, bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, prob_font_scale, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        if type == 0: cnt_mask += 1
        else: cnt_nomask += 1

        color = (0, 211, 255) if type else (0, 255, 0)

        cv2.rectangle(img_cp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

    cv2.line(img_cp, (x, y), (x + len_line, y), (0, 255, 0), 1)
    cv2.putText(img_cp, 'Mask', (x + len_line + pad_x, y + pad_text),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

    cv2.line(img_cp, (x, y + pad_y), (x + len_line, y + pad_y), (0, 211, 255), 1)
    cv2.putText(img_cp, 'No-mask', (x + len_line + pad_x, y + pad_y + pad_text),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
    
    mask_percent = (0 if cnt_mask == 0 else (cnt_mask / (cnt_mask + cnt_nomask))) * 100
    cv2.putText(img_cp, 'Mask percent: {:.0f}%'.format(mask_percent), (x + pad_percent, y + pad_text),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
    return img_cp


def demo_from_file(path, save_out=False, out_dir=None):
    imgs = open(path, 'r').readlines()

    id = 0
    for img_path in imgs:
        img_path = img_path.strip().split(' ')[0]

        bboxs = ssh_detect(img_path)
        img = draw(img_path, bboxs)
        im_show(img)

        if save_out and not out_dir is None:
            id += 1
            cv2.imwrite(out_dir + str(id).zfill(5) + '.jpg', img)

def predict_image(path, saved_path):
    ssd_bboxs = ssd_detect(im_path=path, target_shape=(360, 360))
    ssh_bboxs = ssh_detect(path)
    bboxs = merge_boxes(ssd_bboxs, ssh_bboxs)

    img = draw(path, bboxs)
    # img = draw(path, ssd_bboxs)
    cv2.imwrite(saved_path, img)

def predict_folder(input_f='data/public_test', output_f='data/prediction'):
    for filename in os.listdir(input_f):
        print(filename)
        path = os.path.join(input_f, filename)
        saved_path = os.path.join(output_f, filename[:-4] + '.png')
        predict_image(path, saved_path)

def demo_from_dir(dir, save_out=False, out_dir=None):
    imgs = [
        file
        for file in os.listdir(dir)
        if file.endswith('.jpg')
    ]

    id = 0
    for img_path in imgs:
        img_path = dir + img_path

        bboxs = ssh_detect(img_path)
        img = draw(img_path, bboxs)
        im_show(img)
        
        if save_out and not out_dir is None:
            id += 1
            cv2.imwrite(out_dir + str(id).zfill(5) + '.jpg', img)


def demo_video(video_path=0, save_out=False, out_path='./data/videos/output.avi', visualize=False):
    cap = cv2.VideoCapture(video_path)

    if save_out:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None

    while (cap.isOpened()):
        ret, frame = cap.read()

        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            start_time = time.time()

            # with videos we don't use SSH pyramid option to improve performance
            bboxs = ssh_detect(im=frame)
            frame = draw(None, bboxs, img=frame)

            print("FPS: ", 1.0 / (time.time() - start_time))

            if save_out:
                if out is None:
                    out = cv2.VideoWriter(
                        out_path, fourcc, 20.0,
                        (frame.shape[1], frame.shape[0]))
                out.write(frame)

            if visualize:
                max_size = 1024

                if max(frame.shape[0], frame.shape[1]) > max_size:
                    scale = max_size / max(frame.shape[0], frame.shape[1])
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)

                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if ret == False: break

    cap.release()
    if save_out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # demo_video(net, visualize=True)
    
    # uncomment below to run demo on video
    # demo_video(net, './data/videos/demo3.MOV', save_out=True, visualize=True)
    # path = 'test.jpg'
    # saved_path = 'result.png'
    # predict_image(path, saved_path)
    # predict_folder(output_f='data/ssd_mobilenet_result')
    # predict_folder(output_f='data/ehdetect_mobilenet_result')
    predict_folder(output_f='data/check')
