import pandas as pd
import os
import cv2
from lib.SSH_pytorch.face_detection import ssh_detect
from lib.SSD.face_detection import ssd_detect
from enhance_detection import merge_boxes

# load meta data
meta_path = 'data/train_meta.csv'
df = pd.read_csv(meta_path)
mask_labels = list(df['mask'])

def labels(faces, idx):
    is_mask = mask_labels[idx-1]
    labels_dic = {'mask': [], 'no_mask': []}

    if is_mask == 1:
        labels_dic['mask'] = list(range(len(faces)))
    else:
        for i in range(len(faces)):
            label, conf, _, _, _, _ = faces[i]
            if label == 1 and conf > 0.9: # no_mask with high probability
                labels_dic['no_mask'].append(i)
            # if label == 0 and conf > 0.9:
                # labels_dic['mask'].append(i)
    
    labels_li = ['nan']*len(faces)
    for j in labels_dic['mask']:
        labels_li[j] = 'mask'
    for k in labels_dic['no_mask']:
        labels_li[k] = 'no_mask'

    return labels_li



if __name__ == '__main__':
    img_folder = 'data/train_images'

    filenames = os.listdir(img_folder)
    for i in range(1, len(filenames)+1):
        print(i)
        filename = str(i) + '.jpg'
        path = os.path.join(img_folder, filename)

        img = cv2.imread(path)
        copy_img = img.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ssd_bboxs = ssd_detect(im=img, target_shape=(360, 360), classify=True)
        ssh_bboxs = ssh_detect(im=img, classify=True)
        faces = merge_boxes(ssd_bboxs, ssh_bboxs, classify=True)
        
        labels_li = labels(faces, i)

        # loop over all faces
        for j in range(len(faces)):
            face = faces[j]
            xmin, ymin, xmax, ymax = face[2:]

            # crop and save
            xmin = max(0, int(xmin) - 4)
            xmax = min(copy_img.shape[1], int(xmax) + 4)
            ymin = max(0, int(ymin) - 2)
            ymax = min(copy_img.shape[0], int(ymax) + 2)

            crop_img = copy_img[ymin:ymax, xmin:xmax]
            label_folder = os.path.join('data/train_faces', labels_li[j])
            saved_path = os.path.join(label_folder, filename[:-4] + '_%d.png'%j)
            cv2.imwrite(saved_path, crop_img)