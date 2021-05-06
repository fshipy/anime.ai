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

from dataset.dataset_interface import AnimeDataset
from dataset.dataset_interface import transform

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 2
TRAIN_EPOCH = 50
USE_CUDA = True
splitted = True # whether train and test are splitted
model_used = "alexnet" # "simple"
apply_augmentations = True
compute_mean_std = False # do we want to compute the mean/std of dataset

def compute_ds_mean_std(labels):
    # get mean and standard deviation in the dataset

    all_data, all_data_count = get_data_path('dataset\\dataset_all', labels)
    
    transform = transforms.Compose([transforms.Resize((IMAGE_HEIGHT + 20, IMAGE_WIDTH + 20)), transforms.CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor()])

    dataset = AnimeDataset((IMAGE_HEIGHT, IMAGE_WIDTH), all_data, all_data_count, "TRAIN", transform)

    means = []
    stds = []
    
    for data in dataset:
        img = data[1]
        m = torch.mean(img)
        o = torch.std(img)
        means.append(m)
        stds.append(o)
        #print(m, o)

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))
    print("Means:", means)
    print("stds:", stds)
    print(mean, std)


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
    normalize = transforms.Normalize(mean=[0.65, 0.65, 0.65],
                                     std=[0.26, 0.26, 0.26])
    train_transforms = None
    test_tranforms = None                        
    if apply_augmentations:
        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                # maybe add ColorJitter here
                transforms.ToTensor(),
                normalize,
            ])
        test_tranforms = transforms.Compose([
                transforms.Resize((IMAGE_HEIGHT + 20, IMAGE_WIDTH + 20)),
                transforms.CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
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
    
    print("Mean Accuracy: {:.1f} %".format(sum(all_accuracy) / len(characterNames)))
    print("correct_pred", correct_pred)
    print("total_pred", total_pred)

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

    print("\n[Training Accuracy]")
    _print_accuracy(correct_pred, total_pred, characterNames)

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
    
    print("\n[Validation Accuracy]")
    _print_accuracy(correct_pred, total_pred, characterNames)

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

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("total training images:", len(trainLoader.dataset))
    print("total training batches:", len(trainLoader))

    print("total testing images:", len(testLoader.dataset))
    print("total testing batches:", len(testLoader))
    
    for epoch in range(TRAIN_EPOCH):
    
        train(model, optimizer, trainLoader, criterion, device, epoch, labels, log_interval=50)

        test(model, testLoader, labels, device)
    
    # print(all_data)
    # print(all_data_count)
    predict(model, 'Ruri Gokou_350_0.jpg', labels, device)
    predict(model, 'Shana_370_0.jpg', labels, device)

if __name__ == "__main__":
    main()