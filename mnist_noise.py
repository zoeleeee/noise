import torch
import torch.nn as nn
import torch.nn.functional as F

class AFT_Net(nn.Module):
    def __init__(self, layer_id=0):
        super(AFT_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.layer_id = layer_id
        self.nb_layer = 6

    def set_layer_id(self, layer_id):
        self.layer_id = layer_id
        print(self.layer_id)
    def forward(self, x):
        print(self.layer_id)
        layer_id = self.layer_id
        # for i in range(layer_id, nb_layer)
        while self.nb_layer > layer_id:
            if layer_id == 0:
                x = F.relu(self.conv1(x))
            elif layer_id == 1:
                x = self.pool1(x)
            elif layer_id == 2:
                x = F.relu(self.conv2(x))
            elif layer_id == 3:
                x = self.pool2(x)
            elif layer_id == 4:
                x = x.view(-1, 3136)
                x = F.relu(self.fc1(x))
            elif layer_id == 5:
                x = self.fc2(x)
            layer_id += 1
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 10)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x0 = F.relu(self.conv1(x))
        x1 = self.pool1(x0)
        x2 = F.relu(self.conv2(x1))
        x3 = self.pool2(x2)
        x4 = x3.view(-1, 3136)
        x4 = F.relu(self.fc1(x4))
        x5 = self.fc2(x4)
        # x = self.fc3(x)
        return [x0, x1, x2, x3, x4, x5]

if __name__ == '__main__':
    import sys
    file = sys.argv[-1]
    net = Net()
    net.load_state_dict(torch.load(file))
    aft = AFT_Net()
    aft.load_state_dict(torch.load(file))

    import numpy as np
    data = np.load('data/mnist_data.npy').astype(np.float32) / 255.
    inputs = torch.tensor(data[:1])

    outs = net(inputs)
    print(outs[-1])
    for i,out in enumerate(outs):
        aft.set_layer_id(i+1)
        val = aft(out)
        if torch.sum(torch.abs(val-outs[-1])) != 0:
            print(val, outs[-1]-val)
