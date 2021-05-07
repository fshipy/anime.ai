import torchvision.models as models

def alex_net(num_classes=10):
    return models.AlexNet(num_classes)