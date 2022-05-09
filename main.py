import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from model.simple_custom_net import SimpleAnimeNet
from model.alexnet import alex_net
from model.vgg11 import *
from model.resnet18 import resnet18

from dataset.dataset_interface import AnimeDataset
from dataset.dataset_interface import transform

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4
TRAIN_EPOCH = 100
USE_CUDA = True
splitted = True # whether train and test are splitted

"""
available networks:

simple
alexnet
vgg11
vgg11_bn

"""
model_used = "resnet18"
pretrained = True
apply_augmentations = True
compute_mean_std = False # do we want to compute the mean/std of dataset
lr = 5e-3
momentum = 0.9
weight_decay = 5e-4

def compute_ds_mean_std(labels):
    # get mean and standard deviation in the dataset

    all_data, all_data_count = get_data_path('dataset\\dataset_all', labels)
    print("labels", labels)
    
    transform = transforms.Compose([transforms.Resize((IMAGE_HEIGHT + 20, IMAGE_WIDTH + 20)), transforms.CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor()])

    dataset = AnimeDataset((IMAGE_HEIGHT, IMAGE_WIDTH), all_data, all_data_count, "TRAIN", transform)

    means = []
    means_0 = []
    means_1 = []
    means_2 = []
    stds = []
    stds_0 = []
    stds_1 = []
    stds_2 = []
    
    for data in dataset:
        img = data[1]
        m = torch.mean(img)
        m0 = torch.mean(img[0])
        m1 = torch.mean(img[1])
        m2 = torch.mean(img[2])

        o = torch.std(img)
        o_0 = torch.std(img[0])
        o_1 = torch.std(img[1])
        o_2 = torch.std(img[2])

        means.append(m)
        means_0.append(m0)
        means_1.append(m1)
        means_2.append(m2)
        stds.append(o)
        stds_0.append(o_0)
        stds_1.append(o_1)
        stds_2.append(o_2)
        #print(m, o)

    mean = torch.mean(torch.tensor(means))
    mean0 = torch.mean(torch.tensor(means_0))
    mean1 = torch.mean(torch.tensor(means_1))
    mean2 = torch.mean(torch.tensor(means_2))
    std = torch.mean(torch.tensor(stds))
    std0 = torch.mean(torch.tensor(stds_0))
    std1 = torch.mean(torch.tensor(stds_1))
    std2 = torch.mean(torch.tensor(stds_2))
    print("Means:", means)
    print("stds:", stds)
    print("MEANS:")
    print(mean, mean0, mean1, mean2)
    print("STD:")
    print(std, std0, std1, std2)

    """
    with Others (11 classes)
    MEANS:
    tensor(0.6862) [tensor(0.7168) tensor(0.6650) tensor(0.6770)]
    STD:
    tensor(0.2488) [tensor(0.2401) tensor(0.2476) tensor(0.2352)]

    without Others (10 classes)
    MEANS:
    tensor(0.6902) [tensor(0.7194) tensor(0.6689) tensor(0.6822)]
    STD:
    tensor(0.2482) [tensor(0.2395) tensor(0.2472) tensor(0.2345)]
    """


def read_labels(path):
    label_file = open(path, 'r')
    return [line.rstrip('\n') for line in label_file.readlines()]

def ind_to_label(index, labels):
    return labels[index]

def get_data_path(data_folder, labels):
    all_data = {}
    all_data_count = {}
    for i, label in enumerate(labels):
        files = os.listdir(os.path.join(data_folder, label))
        all_data_count[i] = 0
        all_data[i] = []
        for f in files:
            all_data_count[i] += 1
            all_data[i].append(os.path.join(data_folder, label, f))
    return all_data, all_data_count

def load_dataset(train_data_dict, train_data_count, test_data_dict, test_data_count):

    # tensor(0.6557) tensor(0.2586)
    normalize = transforms.Normalize(mean=[0.7194, 0.6689, 0.6822],
                                     std=[0.2395, 0.2472, 0.2345])
    train_transforms = None
    test_tranforms = None                        
    if apply_augmentations:
        train_transforms = transforms.Compose([
                transforms.Resize((IMAGE_HEIGHT + 20, IMAGE_WIDTH + 20)),
                transforms.RandomCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.1),
                transforms.RandomGrayscale(0.1),
                transforms.ToTensor(),
                normalize,
            ])
        test_tranforms = transforms.Compose([
                transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
                normalize,
            ])
    
    trainSet = AnimeDataset((IMAGE_HEIGHT, IMAGE_WIDTH), train_data_dict, train_data_count, "TRAIN", train_transforms)
    testSet = AnimeDataset((IMAGE_HEIGHT, IMAGE_WIDTH), test_data_dict, test_data_count, "TEST", test_tranforms)

    trainLoader = DataLoader(trainSet, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    testLoader = DataLoader(testSet, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    return trainLoader, testLoader

def split_dataset(all_data, all_data_count, test_ratio=0.2):
    train_data_dict = {}
    train_data_count = {}
    test_data_dict = {}
    test_data_count = {}
    for key in list(all_data_count.keys()):
        all_data_key = all_data[key]
        random.shuffle(all_data_key)
        test_data_count[key] = int(all_data_count[key] * test_ratio)
        train_data_count[key] = all_data_count[key] - test_data_count[key]
        train_data_dict[key] = all_data_key[:train_data_count[key]].copy()
        test_data_dict[key] = all_data_key[train_data_count[key]:].copy()

    # split to train and test set
    return train_data_dict, train_data_count, test_data_dict, test_data_count
    #return all_data.copy(), all_data_count.copy(), all_data.copy(), all_data_count.copy()

def _print_accuracy(correct_pred, total_pred, characterNames):
    all_accuracy = []
    # print accuracy of each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for character {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))
        all_accuracy.append(accuracy)
    
    mean_acc = sum(all_accuracy) / len(characterNames)
    print("Mean Accuracy: {:.1f} %".format(mean_acc))
    print("correct_pred", correct_pred)
    print("total_pred", total_pred)
    return mean_acc

def train(model, optimizer, trainLoader, criterion, device, epoch, characterNames, log_interval):

    model.train()
    running_loss = 0.0
    
    correct_pred = {classname: 0 for classname in characterNames}
    total_pred = {classname: 0 for classname in characterNames}
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        labels, images = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % log_interval == log_interval - 1:    # print every 100 mini-batches
            print('[Epoch %d, Iteration %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_interval))
            running_loss = 0.0
        
        # compute training accuracy
        with torch.no_grad():
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[ind_to_label(label, characterNames)] += 1
                total_pred[ind_to_label(label, characterNames)] += 1

        del labels, images, loss, outputs, predictions # free some memory since their histories may be stored

    print("\n[Training Accuracy]")
    train_acc = _print_accuracy(correct_pred, total_pred, characterNames)
    return train_acc # return the mean

def test(model, testLoader, characterNames, device):

    model.eval()
    
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in characterNames}
    total_pred = {classname: 0 for classname in characterNames}
    # again no gradients needed
    with torch.no_grad():
        for data in testLoader:
            labels, images = data[0].to(device), data[1].to(device)
            outputs = model(images)
            
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[ind_to_label(label, characterNames)] += 1
                total_pred[ind_to_label(label, characterNames)] += 1

            del labels, images, outputs, predictions # free some memory since their histories may be stored
    
    print("\n[Validation Accuracy]")
    val_acc = _print_accuracy(correct_pred, total_pred, characterNames)
    return val_acc # return the mean

def predict(model, imagePath, characterNames, device):
    model.eval()
    with torch.no_grad():
        image = Image.open(imagePath, mode='r')
        image = image.convert('RGB')
        _, image = transform(None, image, (IMAGE_HEIGHT, IMAGE_WIDTH), split="TEST")
        image = image.to(device)
        image = torch.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
        outputs = model(image)
        _, predictions = torch.max(outputs, 1)
        prediction = predictions[0]
        print("Predict", imagePath, "is", ind_to_label(int(prediction), characterNames))
        return prediction

# save current state to disk
def save_checkpoint(save_path, model, optimizer, epoch, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, save_path)

# load states from disk
def load_checkpoint(NetClass, OptimizerClass, load_path):
    model = NetClass()
    optimizer = OptimizerClass(model.parameters(), lr=0.001, momentum=0.9)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss

def main():
    labels = read_labels('dataset\\labels.txt')
    
    if splitted:
        train_data_dict, train_data_count = get_data_path('dataset\\train', labels)
        test_data_dict, test_data_count = get_data_path('dataset\\test', labels)
    else:
        all_data, all_data_count = get_data_path('dataset\\cropped', labels)
        train_data_dict, train_data_count, test_data_dict, test_data_count = split_dataset(all_data, all_data_count)
    
    if compute_mean_std:
        compute_ds_mean_std(labels)
    
    trainLoader, testLoader = load_dataset(train_data_dict, train_data_count, test_data_dict, test_data_count)

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    print("Device using:", device)

    if model_used == "alexnet":
        model = alex_net(len(labels))
    elif model_used == "simple":
        model = SimpleAnimeNet(len(labels))
    elif model_used == "vgg11":
        model = vgg11(len(labels))
    elif model_used == "vgg11_bn":
        model = vgg11_bn(len(labels)) 
    elif model_used == "resnet18":
        model = resnet18(len(labels), pretrained) 

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCH, eta_min=1e-6)

    print("total training images:", len(trainLoader.dataset))
    print("total training batches:", len(trainLoader))

    print("total testing images:", len(testLoader.dataset))
    print("total testing batches:", len(testLoader))
    
    for epoch in range(TRAIN_EPOCH):
    
        train(model, optimizer, trainLoader, criterion, device, epoch, labels, log_interval=50)

        test_acc = test(model, testLoader, labels, device)
        scheduler.step() # maybe we can try average training loss / average validation loss?
        save_checkpoint("anime_checkpoint.pt", model, optimizer, epoch, criterion)
    test(model, testLoader, labels, device)

if __name__ == "__main__":
    main()