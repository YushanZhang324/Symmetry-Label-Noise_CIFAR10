'''
8-layer network in pytorch

'''
import torch
import torch.nn as nn
import torch.nn.functional as F




class EightLayerNet(nn.Module):
    def __init__(self):
        super(EightLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)   
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv5 = nn.Conv2d(128, 196, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(196)
        self.conv6 = nn.Conv2d(196, 196, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(196)   
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fc1 = nn.Linear(3136, 256)
        self.classifier = nn.Linear(256, 10)



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool1(out)
        
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool2(out)
        
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = self.pool3(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.classifier(out)
        return out

def test():
    net = EightLayerNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    