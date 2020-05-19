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

def loss_func(preds, target, noise):
    ce = F.cross_entropy(preds, target)
    
    real = torch.sum(preds*target, 1)
    other = torch.max((1-target)*preds-target*1000, 1)
    # dp_robust = tf.max(0.0, real-other+)
    cw = torch.max(0.0, real-other)

    dist = torch.norm(noise)
    return -1*dist + cw

def train(net, layer_id, aft, rnd, trainloader, path, nb_epoch=100):
    optimizer = optim.Adam(rnd.parameters(), lr=0.001, momentum=0.9)
    
    net.eval()
    aft.eval()
    aft.set_layer_id(layer_id+1)
    
    for epoch in range(nb_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if torch.cuda.is_available():
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outs = net(inputs)
            noises = rnd(inputs)
            preds = aft(inputs[layer_id]+noises[layer_id])
            loss = loss_func(preds, labels, noise[layer_id])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d epoch] loss: %.3f' %
                      (epoch + 1, running_loss / 2000))
        running_loss = 0.0

    torch.save(rnd.state_dict(), path)
    print('Finished Training')


if __name__ == '__main__':
    import sys
    file = sys.argv[-1]
    net = Net()
    net.load_state_dict(torch.load(file))
    aft = AFT_Net()
    aft.load_state_dict(torch.load(file))
    rnd = Net()

    import numpy as np
    data = np.load('data/mnist_data.npy').astype(np.float32) / 255.
    labs = np.load('data/mnist_labels.npy').astyep(np.float32) / 255.
    labels = np.zeros((len(labs), np.max(labs)+1))
    for i, l in enumerate(labels):
        l[labs[i]] = 1
    print(np.sum(labels))
    from torch.utils import data
    my_dataset = data.TensorDataset(torch.tensor(data[:60000]), torch.tensor(labels[:60000].astype(np.int)))
    trainloader = data.DataLoader(my_dataset, batch_size = 64, shuffle = True)
    # testloader = 

    for i in range(6):
        train(net, i, aft, rnd, trainloader, 'rnd_'+file)

