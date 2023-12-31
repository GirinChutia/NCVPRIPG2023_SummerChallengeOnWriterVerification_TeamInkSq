import os
import glob
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


class Dataset(torch.utils.data.IterableDataset):

    # Ref : https://github.com/sohaib023/siamese-pytorch/blob/master/libs/dataset.py

    def __init__(self, path, shuffle_pairs=True, augment=False):
        """
        Create an iterable dataset from a directory containing sub-directories of 
        entities with their images contained inside each sub-directory.
            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_pairs (boolean):    Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.
            where b = batch size
            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        """
        self.path = path

        self.feed_shape = (3, 200,2600)
        self.shuffle_pairs = shuffle_pairs

        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.05
                    ),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.Resize(self.feed_shape[1:], antialias=True),
                ]
            )
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.Resize(self.feed_shape[1:], antialias=True),
                ]
            )

        self.create_pairs()

    def create_pairs(self):
        """
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        """

        self.image_paths = glob.glob(os.path.join(self.path, "*/*.jpg"))
        self.image_classes = []
        self.class_indices = {}

        for image_path in self.image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            self.class_indices[image_class].append(self.image_paths.index(image_path))

        self.indices1 = np.arange(len(self.image_paths))

        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(1)

        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        self.indices2 = []

        for i, pos in zip(self.indices1, select_pos_pair):
            class1 = self.image_classes[i]
            if pos:
                class2 = class1
            else:
                class2 = np.random.choice(
                    list(set(self.class_indices.keys()) - {class1})
                )
            idx2 = np.random.choice(self.class_indices[class2])
            self.indices2.append(idx2)
        self.indices2 = np.array(self.indices2)

    def __iter__(self):
        self.create_pairs()

        for idx, idx2 in zip(self.indices1, self.indices2):

            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]

            class1 = self.image_classes[idx]
            class2 = self.image_classes[idx2]

            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()
            yield (image1, image2), torch.FloatTensor([class1 == class2]), (
                class1,
                class2,
            ), (image_path1, image_path2)

    def __len__(self):
        return len(self.image_paths)


def imshow(img, text=None):
    npimg = img.numpy() * 255
    plt.figure(figsize=(20, 10))
    plt.axis("off")
    if text:
        plt.text(
            75,
            8,
            text,
            style="italic",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
        )
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
