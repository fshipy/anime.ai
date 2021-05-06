import torchvision.models as models

def alex_net(num_classes):
    return models.AlexNet(num_classes)