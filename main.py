import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from model.simple_custom_net import SimpleAnimeNet
from dataset.dataset_interface import AnimeDataset
from dataset.dataset_interface import transform
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 2
TRAIN_EPOCH = 50
USE_CUDA = True

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
    trainSet = AnimeDataset((IMAGE_HEIGHT, IMAGE_WIDTH), train_data_dict, train_data_count, "TRAIN")
    testSet = AnimeDataset((IMAGE_HEIGHT, IMAGE_WIDTH), test_data_dict, test_data_count, "TEST")

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
    
def main():
    labels = read_labels('dataset\\labels.txt')
    all_data, all_data_count = get_data_path('dataset\\cropped', labels)
    train_data_dict, train_data_count, test_data_dict, test_data_count = split_dataset(all_data, all_data_count)
    
    trainLoader, testLoader = load_dataset(train_data_dict, train_data_count, test_data_dict, test_data_count)

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    print("Device using:", device)

    model = SimpleAnimeNet(len(list(all_data_count.keys())))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("total training images:", len(trainLoader.dataset))
    print("total training batches:", len(trainLoader))

    print("total testing images:", len(testLoader.dataset))
    print("total training batches:", len(testLoader))
    
    for epoch in range(TRAIN_EPOCH):
    
        train(model, optimizer, trainLoader, criterion, device, epoch, labels, log_interval=50)

        test(model, testLoader, labels, device)
    
    # print(all_data)
    # print(all_data_count)
    predict(model, 'Ruri Gokou_350_0.jpg', labels, device)
    predict(model, 'Shana_370_0.jpg', labels, device)
if __name__ == "__main__":
    main()