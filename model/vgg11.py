import torchvision.models as models
import torch.nn as nn

def vgg11_bn(num_classes=10):
    model = models.vgg11_bn(pretrained=False)
    model.classifier[6] = nn.Linear(4096,num_classes)
    return model

def vgg11(num_classes=10):
    model = models.vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096,num_classes)
    return model