from dataset import MyData, load_med_img
from model import resnet101
from fintune_model import SingleResFc
from fintune_model import ResFc
import torch
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from eval import Rmse, r2

epochs = 50
batch_size = 16
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# net = ResFc(4)
# net = SingleResFc(5)
net = SingleResFc(1)
net.to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_transform = transforms.Compose([
    # transforms.Resize(512),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = MyData(root='',
                    datatxt='',
                    transform=train_transform)
val_data = MyData(root='',
                  datatxt='',
                  transform=val_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size)

length_tr = len(train_loader)
length_val = len(val_loader)


def train_img(epochs, model_name):

    for epoch in range(epochs):
        running_loss = 0.0
        rmse1 = 0.0
        R21 = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            rmse1 += Rmse(labels, outputs)
            R21 += r2(labels, outputs)
            running_loss += loss.item()

        net.eval()
        val_loss = 0.0
        val_r2, val_rmse = 0.0, 0.0
        for i, data in enumerate(val_loader):
            val_img, val_label = data
            val_img, val_label = val_img.to(device), val_label.to(device)
            with torch.no_grad():
                # optimizer.zero_grad()
                outputs = net(val_img)
                loss = criterion(outputs, val_label)
                val_loss += loss.item()

                val_r2 += r2(val_label, outputs)
                val_rmse += Rmse(val_label, outputs)

        print('epoch{}: train_loss:{:.3f}\t train_rmse:{:.5f}\t train_r2:{:.5f}\t'
              'val_loss:{:.4f}\t val_rmse:{:.4f}\t val_r2:{:.4f}\t \n'.
              format(epoch, running_loss/length_tr, rmse1/length_tr, R21/length_tr,
                     val_loss / length_val, val_rmse / length_val, val_r2 / length_val,))

    torch.save(net.state_dict(), model_name)


def train_one(epochs, model_name):
    label = ''
    image = ''
    x, y = load_med_img(image, label)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=111)

    # x, y = torch.tensor(x), torch.tensor(y)
    for epoch in range(epochs):

        outputs = net(train_x)
        loss = criterion(train_y, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        R2 = r2(train_y, outputs)
        rmse = Rmse(train_y, outputs)

        if (epoch + 1) % 2 == 0:
            print('Epoch[{}/{}], loss: {:.6f}, R2:{:.3f}, RMSE:{:.3f}'.format(epoch + 1, epochs, loss.data,
                                                                              R2, rmse))
    torch.save(net.state_dict(), model_name)


def test(PATH):
    test_data = MyData(root='',
                       datatxt='',
                       transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    rmse = 0.0
    R2 = 0.0
    net.eval()
    net.load_state_dict(torch.load(PATH))
    n = len(test_loader) - 1
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels[0])
            outputs = net(images)

            rmse += Rmse(labels, outputs)
            R2 += r2(labels, outputs)
        print('[-------------%2d]  test_rmse()：%.3f  test_R2:%.3f' % (n, rmse / n, R2 / n))



if __name__ == '__main__':

    train_img(200, model_name='')

