import torch
import os
import random
import numpy as np
import torchvision.transforms.functional as FT
from PIL import Image
from torch.utils.data import Dataset

def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image

def transform(label, image, input_shape, split):
    new_image = image
    width, height = image.size
    if split == 'TRAIN':
        # flip the image
        new_image = photometric_distort(new_image)

        # if random.random() < 0.5: # zoom in
        #     cropped_offset = 25
        #     new_image = FT.center_crop(new_image, (height - cropped_offset, width - cropped_offset))

        # if random.random() < 0.5: # random crop and zoom out
        #     new_image = FT.RandomCrop(6, padding=1)

        # if random.random() < 0.5: # rotation
        #     new_image = FT.RandomRotation(10)

        if random.random() < 0.5: # flip horizontally
            new_image = FT.hflip(new_image)
        # add more augmentation here
    
    new_image = FT.resize(image, input_shape)
    
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    #new_image = np.subtract(np.divide(new_image, 255), 0.5)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    new_image = FT.normalize(new_image, mean=mean, std=std)
    return label, new_image

class AnimeDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, input_shape, data_path, data_count, split, transform=None):
        """
        :param input_shape (tuple): (height, width) of a sample image
        :param data_path (dict): dictionary that maps label to list of image path
        :param data_count (dict): dictionary that maps label to number of samples
        :param split (str): test or train
        """

        def dict_to_ltuple(data_dict):
            # assume data_dict maps label to list of datapath
            all_data_tuple = []
            for key in list(data_dict.keys()):
                for p in data_dict[key]:
                    all_data_tuple.append((key, p))
            return all_data_tuple

        #self.data_path = data_path
        assert split.upper() in ("TRAIN", "TEST")
        self.split = split.upper()
        self.data_count = data_count
        self.all_data = dict_to_ltuple(data_path)
        self.input_shape = input_shape
        self.transform = transform
        c = 0
        for v in list(data_count.values()):
            c += v
        assert len(self.all_data) == c

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.all_data[i][1], mode='r')
        image = image.convert('RGB')
        label = torch.LongTensor([self.all_data[i][0]])[0]
        # Apply transformations
        if self.transform: 
            image = self.transform(image)
        else: # apply our default (simple) augmentations
            label, image = transform(label, image, self.input_shape, split=self.split)
        return label, image, self.all_data[i][1]

    def __len__(self):
        return len(self.all_data)