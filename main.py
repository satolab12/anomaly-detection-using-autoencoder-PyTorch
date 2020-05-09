###############libraryの読み込み
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pylab
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

mount_dir = "./drive/My Drive/"

#######
class Mnisttox(Dataset):
    def __init__(self, datasets ,label):
        self.dataset = datasets
        self.label = label

    def __len__(self):
        return int(len(self.dataset)/10)

    def __getitem__(self, index):
        i = 0
        while(True):
            img,label = self.dataset[index+i]
            if label == self.label:
                return img, label
            i += 1

class Mnisttoxy(Dataset):
    def __init__(self, datasets ,label):
        self.dataset = datasets
        self.label = label
        
    def __len__(self):
        return int((len(self.dataset)/10)*2)

    def __getitem__(self, index):
        i = 0
        while(True):
            img,label = self.dataset[index+i]
            if label == self.label[0]or label == self.label[1]:
                return img, label
            i += 1

class Autoencoder(nn.Module):
    def __init__(self,z_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128 , z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat

##################パラメータ############
z_dim = 28*28 #2 #16 
batch_size = 16
num_epochs = 10
learning_rate = 0.0002 
cuda = True

model = Autoencoder(z_dim)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)

if cuda:
    model.cuda()
  
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))  # [0,1] => [-1,1]
])

train_dataset = MNIST(mount_dir + 'data', download=True, transform=img_transform)#手書き数字
train_1 = Mnisttox(train_dataset,1)
train_loader = DataLoader(train_1, batch_size=batch_size, shuffle=True)
losses = np.zeros(num_epochs*len(train_loader))

i = 0
for epoch in range(num_epochs):   
    for data in train_loader:
        img, label = data
        x = img.view(img.size(0), -1)

        if cuda:
            x = Variable(x).cuda()
        else:
            x = Variable(x)

        xhat = model(x)

        # 出力画像（再構成画像）と入力画像の間でlossを計算
        loss = mse_loss(xhat, x)
        losses[i] = loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    
    fig , ax = plt.subplots()
    pylab.xlim(0, num_epochs)
    pylab.ylim(0, 1)
    x = np.linspace(0,num_epochs,len(losses))
    ax.plot(x, losses, label='loss')
    ax.legend()
    plt.savefig(os.path.join(mount_dir + "experiments/save", 'loss.pdf'))
    plt.close()


    print('epoch [{}/{}], loss: {:.4f}'.format(
        epoch + 1,
        num_epochs,
        loss))

########################################
test_dataset = MNIST(mount_dir + 'data', train=False,download=True, transform=img_transform)
test_1_9 = Mnisttoxy(test_dataset,[1,9])
test_loader = DataLoader(test_1_9, batch_size=len(test_1_9), shuffle=True)

data = torch.zeros(6,28*28)
j = 0
for img ,label in (test_loader):
    x = img.view(img.size(0), -1)
    data = x
    
    if cuda:
        data = Variable(data).cuda()
    else:
        data = Variable(data)

    xhat = model(data)
    data = data.cpu().detach().numpy()
    xhat = xhat.cpu().detach().numpy()
    data = data/2 + 0.5
    xhat = xhat/2 + 0.5
    
# サンプル画像表示
n = 6
plt.figure(figsize=(12, 6))
for i in range(n):
    # テスト画像を表示
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 出力画像を表示
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(xhat[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 入出力の差分画像を計算
    diff_img = np.abs((data[i] - xhat[i]))

    # 入出力の差分数値を計算
    diff = np.sum(np.abs(data[i] - xhat[i]))

    # 差分画像と差分数値の表示
    ax = plt.subplot(3, n, i + 1 + n * 2)
    plt.imshow(diff_img.reshape(28, 28),cmap="jet")
    #plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_xlabel('score = ' + str(diff))

plt.savefig(mount_dir + "experiments/save/result.png")
plt.show()
plt.close()
