from .Net import Resnet34
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2

import pathlib
import os.path as osp

class Controller(object):
    def __init__(self, model_path=None, gpu_id=0):
        # set model path
        if model_path is None:
            curr_dir = pathlib.Path(__file__).parent.absolute()
            model_path = osp.join(curr_dir, '../assets/levit_128s.pth')

        # set distribution class list
        distribution_classes = [
            '0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','I','J',
            'K','L','M','N','O','P','Q','R','S','T',
            'U','V','W','X','Y','Z'
        ]
        n_classes = len(distribution_classes)
        self.distribution_classes = distribution_classes
        self.device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        # 模型定义和加载
        self.model = Resnet34(n_classes=n_classes,pretrained=False) 
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def process(group):
        """
        对一组图片进行预处理 
        """
        img_list = []
        resize = transforms.Resize([224,224])
        toTensor = transforms.ToTensor()
        for img in group:
            if img.ndim!=2:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转化为灰度图
            blur = cv2.GaussianBlur(img,(5,5),0)
            _,thImg = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  
            mask = cv2.erode(thImg.astype('uint8'), kernel=np.ones((3,3)))
            #resize and normalize
            mask = Image.fromarray(mask)
            mask = toTensor(resize(mask)) 
            img_list.append(mask.cpu().numpy())
        return torch.tensor(img_list) 
    

    def get_pred_str(self,group):
        img_list = self.process(group)
        if self.device.type=="cuda":
            img_list = img_list.cuda()
        output = self.model(img_list)
        result = ""
        for index in output:
            result += self.distribution_classes[index] 
        return result

    infer = get_pred_str
