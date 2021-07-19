import torch.nn as nn
import timm

class Resnet34(nn.Module):
    def __init__(self,n_classes,pretrained = False):
        super(Resnet34,self).__init__()
        self.model = timm.create_model("levit_128s",pretrained=pretrained,in_chans =1,num_classes=n_classes)
    def forward(self,x):
        x = self.model(x)
        x = x.argmax(dim=1)
        return x