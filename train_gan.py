from PIL import Image 
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from model.simple_custom_net import SimpleAnimeNet
from model.alexnet import alex_net
from model.vgg11 import *
from model.gan import *
from main import read_labels, test, load_dataset, get_data_path

USE_CUDA = True
lr = 0.0002
beta1 = 0.5
input_shape = 100
batch_size = 16
num_classes = 10
log_interval = 100
save_interval = 1000
pretrained_models_path = ["anime_checkpoint_alex_class10.pt", "anime_checkpoint_vgg11_class10.pt"]
pretrained_models_class = [alex_net, vgg11]
train_iteration = 20000
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

def make_input(device):
    noise = torch.randn(batch_size, input_shape - num_classes, 1, 1, device=device)
    label = torch.randint(0, num_classes, (batch_size, 1, 1), device=device)
    one_hot_label = F.one_hot(label, num_classes=num_classes)
    one_hot_label = torch.reshape(one_hot_label, (batch_size, num_classes, 1, 1))
    input_to_model = torch.cat((one_hot_label, noise), 1)
    return input_to_model, torch.flatten(label)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def show_images(img_list, save_name):
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list,(1,2,0)))
    plt.savefig(save_name)
    plt.show()

def generate_images(iteration, input, label, gan_model, pretrained_classifiers):
    gan_model.eval()
    unorm = UnNormalize(mean=(0.7194, 0.6689, 0.6822), std=(0.2395, 0.2472, 0.2345))
    with torch.no_grad():
        output = gan_model(input)
        for pretrained_classifier in pretrained_classifiers:
            classified_by_model = pretrained_classifier(output)
            _, classified_by_model = torch.max(classified_by_model, 1)
            print("classified_by_model", classified_by_model)
        print("ground_truth", label)
        output = output.detach().cpu()
        print(output.shape)

        img_list_normalized = vutils.make_grid(output, padding=2, normalize=True)
        img_list_unnormalized = vutils.make_grid(output, padding=2, normalize=False)
        show_images(img_list_normalized, "normalized_" + str(iteration) + ".png")
        show_images(img_list_unnormalized, "unnormalized_" + str(iteration) + ".png")

        for i in range(output.shape[0]):
            image = output[i]
            image = unorm(image) * 255
            print(image.shape)
            image = image.permute(2, 1, 0).byte().numpy()
            print(image.shape)
            # load the image (creating a random image as an example)
            pil_image = FT.to_pil_image(image)
            pil_image.save("gan_iteration_" + str(iteration) + "_" + str(i) + ".jpg")
            del image

def train(gan_model, pretrained_classifiers, optimizer, criterion, iterations, device):
    running_loss = 0.0
    # used to check how the generator is doing
    fixed_input, fixed_label = make_input(device)
    for pretrained_classifier in pretrained_classifiers:
        pretrained_classifier.eval()
    for i in range(iterations):
        optimizer.zero_grad()
        # make random noise and label input
        input_to_model, label = make_input(device)
        # feed input to the model
        fake_out = gan_model(input_to_model)
        # use classifier model to benchmark the feeding input
        total_loss = None
        for pretrained_classifier in pretrained_classifiers:
            classified = pretrained_classifier(fake_out)
            # compute the loss
            loss = criterion(classified, label)
            if total_loss is not None:
                total_loss += loss
            else:
                total_loss = loss
    
        total_loss.backward()
        optimizer.step()
        # print statistics
        running_loss += total_loss.item()
        if i % log_interval == log_interval - 1:    # print every <log_interval> mini-batches
            print('[Iteration %5d] loss: %.3f' %
                  (i + 1, running_loss / log_interval))
            running_loss = 0.0

        if i % save_interval == save_interval - 1: # save generated image to check how the generator is doing
            generate_images(i, fixed_input, fixed_label, gan_model, pretrained_classifiers)
            gan_model.train()
        del fake_out, classified, loss, input_to_model, label

def evaluate_classifier(pretrained_classifier, device):
    labels = read_labels('dataset/labels.txt')
    #all_data_dict, all_data_count = get_data_path('dataset\\dataset_all', labels)
    all_data_dict, all_data_count = get_data_path('dataset/test', labels)

    print("labels are", labels)
    print("all_data_count", all_data_count)
    _, test_dataloader = load_dataset(all_data_dict, all_data_count, all_data_dict, all_data_count)
    test(pretrained_classifier, test_dataloader, labels, device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    print("device using", device)
    # load the classifer
    pretrained_classifiers = []
    for pretrained_model_class, pretrained_model_path in zip(pretrained_models_class, pretrained_models_path):
        pretrained_classifier, _, _, _  = load_checkpoint(pretrained_model_class, optim.SGD, pretrained_model_path)
        pretrained_classifier.to(device)
        pretrained_classifiers.append(pretrained_classifier)
    
    #pretrained_classifier, _, _, _  = load_checkpoint(pretrained_model_class, optim.SGD, pretrained_model_path)
    # evaluate the classifer performance
    # for pretrained_classifier in pretrained_classifiers:
    #     evaluate_classifier(pretrained_classifier, device)
    # init the generative model
    gan_model = Generator()
    gan_model.to(device)
    gan_model.apply(weights_init)
    optimizer = optim.Adam(gan_model.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.CrossEntropyLoss()
    train(gan_model, pretrained_classifiers, optimizer, criterion, iterations=train_iteration, device=device)

main()
