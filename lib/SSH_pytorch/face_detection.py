import argparse
import os
import numpy as np
# from model.dataset.factory import get_imdb
from lib.SSH_pytorch.model.utils.config import cfg
from lib.SSH_pytorch.model.roi_data_layer.layer import RoIDataLayer
import torch
from lib.SSH_pytorch.model.SSH import SSH
from lib.SSH_pytorch.model.network import save_check_point, load_check_point
import cv2
import torch.optim as optim

from lib.SSH_pytorch.model.nms.nms_wrapper import nms
from lib.SSH_pytorch.model.utils.test_utils import _get_image_blob, _compute_scaling_factor, visusalize_detections


thresh = 0.5
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

saved_model_path = 'lib/SSH_pytorch/check_point/check_point.zip'
assert os.path.isfile(saved_model_path), 'Pretrained model not found'

net = SSH(vgg16_image_net=False)

if (os.path.isfile(saved_model_path)):
    check_point = load_check_point(saved_model_path)
    net.load_state_dict(check_point['model_state_dict'])
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

net.to(device)
net.eval()



def ssh_detect(im_path=None, im = None, classify=False, thresh=0.5):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image, read by open-cv, bgr
    '''
    if im is None:
        im = cv2.imread(im_path)

    with torch.no_grad():

        im_scale = _compute_scaling_factor(im.shape, cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE)
        im_blob = _get_image_blob(im, [im_scale])[0]

        im_info = np.array([[im_blob['data'].shape[2], im_blob['data'].shape[3], im_scale]])
        im_data = im_blob['data']

        im_info = torch.from_numpy(im_info).to(device)
        im_data = torch.from_numpy(im_data).to(device)

        batch_size = im_data.size()[0]
        ssh_rois = net(im_data, im_info)

        inds = (ssh_rois[:, :, 4] > thresh)
        # inds=inds.unsqueeze(2).expand(batch_size,inds.size()[1],5)
        #
        # ssh_roi_keep = ssh_rois[inds].view(batch_size,-1,5)
        ssh_roi_keep = ssh_rois[:, inds[0], :]
        # unscale back
        ssh_roi_keep[:, :, 0:4] /= im_scale

        for i in range(batch_size):
            ssh_roi_single = ssh_roi_keep[i].cpu().numpy()
            nms_keep = nms(ssh_roi_single, cfg.TEST.RPN_NMS_THRESH)
            cls_dets_single = ssh_roi_single[nms_keep, :]
        
    if classify == True:
        cls_dets_single = cls_dets_single.tolist()
        cls_dets_single = [[-1, -1, xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax, _ in cls_dets_single]

    return cls_dets_single

if __name__ == '__main__':
    img = cv2.imread('15.jpg')
    boxes = ssh_detect(img)
    print(boxes)