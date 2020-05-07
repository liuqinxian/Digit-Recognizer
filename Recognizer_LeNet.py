import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np     

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    transWay = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    trainSet = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transWay
    )
    trainLoader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    testSet = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transWay
    )
    testLoader = torch.utils.data.DataLoader(
        testSet,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    net = LeNet().cuda()
    print('net has been created')

    epochs = 10
    learnRate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learnRate, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%2d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('net has been trained')

    total = 0
    right = 0
    accuracy = 0.0
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net.forward(inputs)
        _, result = torch.max(outputs.data, 1)
        total += labels.size(0)
        right += (result == labels).sum()
    print('Accuracy : %d %%' % (100 * right / total))

    torch.save(net.cpu().state_dict(), './model/LeNet.pth') 
    print('net has been saved')


