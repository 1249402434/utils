from torch import nn
from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
from torch.autograd import Variable

torch.manual_seed(2019)
np.random.seed(2019)

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST('./data/', train=True, transform=transforms.ToTensor(), download=False)
test_data = datasets.MNIST('./data/', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)


class AutoEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super(AutoEncoder, self).__init__()
        self.input_layer = nn.Linear(784, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 784)

    def forward(self, x):
        hidden_out = self.input_layer(x)
        output = self.output_layer(hidden_out)

        return output


model = AutoEncoder(128)
optimizer = SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

model.train()

for e in range(100):
    if e >= 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01

    for X, y in train_loader:
        X = X.reshape(X.shape[0], -1)
        X = Variable(X)
        pred = model(X)
        loss = criterion(pred, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (e + 1) % 10 == 0:
        print("epoch %d loss: %f" % ((e + 1), loss.item()))


def show_image(img, ax):
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))


for X, y in test_loader:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    show_image(torchvision.utils.make_grid(X), ax1)
    pred = model(X)
    pred = pred.reshape(X.shape)
    pred = pred.detach()
    show_image(torchvision.utils.make_grid(pred), ax2)
    plt.show()
