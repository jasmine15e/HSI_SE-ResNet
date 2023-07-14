from dataset import MyData, load_med_img
from model import resnet101
from fintune_model import ResFc, SingleResFc
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from eval import Rmse, r2
import csv


def test(PATH):
    test_data = MyData(root='./data/test/',
                       datatxt='./test.txt',
                       transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    rmse = 0.0
    R2 = 0.0
    net.load_state_dict(torch.load(PATH))
    net.eval()
    n = len(test_loader)
    # res = []
    # y_true = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            a = abs(outputs.cpu().numpy())
            # csv_write.writerow([a[0][0], a[0][1], a[0][2], a[0][3], a[0][4]])
            # y_true.append(labels)
            # res.append(outputs)
            rmse += Rmse(labels, outputs)
            R2 += r2(labels, outputs)
        print('[-------------%2d]  test_rmse()：%.3f  test_R2:%.3f' % (n, rmse / n, R2 / n))


if __name__ == '__main__':
    #  开始预测
    f = open('result_test.csv', 'w', newline='')
    csv_write = csv.writer(f)
    csv_write.writerow(
        ['Salvianolic_pred', 'Dihydrotanshinone_pred', 'Cryptotanshinone_pred', 'Tanshinone_pred', 'Moisture_pred'])
    epochs = 50
    batch_size = 8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = SingleResFc(5)
    # net = resnet101()
    net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    test('')
    f.close()
