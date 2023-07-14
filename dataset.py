import cv2
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os, random, glob
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch


class MyData(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        """
            words: is label and img`s name
        """
        super(MyData, self).__init__()
        file_txt = open(datatxt, 'r')
        imgs = []
        for line in file_txt:
            line = line.rstrip()
            words = line.split(',')
            # print(words)
            # print(words[2:-2])
            img_name = words[2] + '_ref.jpg'
            # words[3:-2]
            imgs.append((img_name, words[7]))

        self.imgs = imgs
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        name, label = self.imgs[index]
        img = Image.open(self.root + name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # label = [float(num) for num in label]
        label = float(label)
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


def load_med_img(img_path, label_path, col=None):
    if col is None:
        col = ['Salvianolic', 'Dihydrotanshinone', 'Cryptotanshinone', 'Tanshinone', 'Moisture']
    df = pd.read_csv(label_path)

    X, Y = [], []
    for i, data in enumerate(df['num']):
        images = img_path + data + '_ref.jpg'
        try:
            img = cv2.imread(images, 0)
            img = cv2.resize(img, (256, 256))
            img = img / 225.0
        except:
            print(images)
            continue
        y = df.loc[i, col[0]]
        X.append(img)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == '__main__':
    label = ''
    image = ''
    x, y = load_med_img(image, label)
    print(x.shape, y.shape)
