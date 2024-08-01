#contains all NCA models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 

class MaxNCA(nn.Module):
    #Classification model NaxNCA - Approach 3
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(MaxNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,13)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        #perceptive function, outputs perception vector
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        #Update function
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        #Forward function, applies k NCA update steps leaving input channels unchanged
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        max=F.adaptive_max_pool2d(x.permute(0, 3, 1, 2), (1, 1))
        max = max.view(max.size(0), -1)
        out=self.fc2(max)
        out = F.relu(out)
        out =self.fc3(out)
        
        return out,x


class SegNCA(nn.Module):
    #Segmentation model
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(SegNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        out=x[...,3]
        
        return out,x




class SimpleNCA(nn.Module):
    #Classification Model SimpleNCA - Approach 1
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(SimpleNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        out = x.mean([1,2])
        out=out[...,self.input_channels:self.input_channels+13]
        
        return out,x
    

class ConvNCA(nn.Module):
    # Classification Model ConvNCA - Approach 2
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(ConvNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,13)
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc2 = nn.Linear(256 * 4 * 4, 512)
        self.fc3 = nn.Linear(512, 13)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        feature_map=x
        x=x.permute(0, 3, 1, 2)
        x=self.pool(x)
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x)))
        x=torch.flatten(x,1,-1)
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out,feature_map















### other models I experimented with that didnt make it into the thesis

class MIL(nn.Module):
    def __init__(self):
        super(MIL, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # attention
        self.class_count=5
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.class_count)
        )

    def forward(self, x):
        ft = x.squeeze(0)
        # calculate attention
        att_raw = self.attention(ft)
        att_raw = torch.transpose(att_raw, 1, 0)
        # Softmax + Matrix multiplication
        att_softmax = F.softmax(att_raw, dim=1)
        bag_features = torch.mm(att_softmax, ft)
        # final classification
        prediction = self.classifier(bag_features)
        return prediction, att_raw, att_softmax, bag_features



class SuperNCA(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(SuperNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,13)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        mean = x.mean([1,2])
        max=F.adaptive_max_pool2d(x.permute(0, 3, 1, 2), (1, 1))
        max = max.view(max.size(0), -1)
        out= torch.cat((mean[:,:64],max[:,64:]),1)
        out=self.fc2(out)
        out = F.relu(out)
        out =self.fc3(out)
        
        return out,x

class BigNCA(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(BigNCA, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p2 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.fc0 = nn.Linear(channel_n*4, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,11)
        with torch.no_grad():
            self.fc1.weight.zero_()
        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        z3 = self.p2(x)
        y = torch.cat((x,z1,z2,z3),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)
        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)
        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        out = self.fc2(x.mean([1,2]))
        out = F.relu(out)
        out =self.fc3(out)
        return out,x
    
class FcNCA(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(FcNCA, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.fc0 = nn.Linear(channel_n*9, 256)
        self.fc00 = nn.Linear(256,hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,11)
        with torch.no_grad():
            self.fc1.weight.zero_()
        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        k1=np.array([[1,0,0],[0,0,0],[0,0,0]])
        k2=np.array([[0,1,0],[0,0,0],[0,0,0]])
        k3=np.array([[0,0,1],[0,0,0],[0,0,0]])
        k4=np.array([[0,0,0],[1,0,0],[0,0,0]])
        k5=np.array([[0,0,0],[0,0,1],[0,0,0]])
        k6=np.array([[0,0,0],[0,0,0],[1,0,0]])
        k7=np.array([[0,0,0],[0,0,0],[0,1,0]])
        k8=np.array([[0,0,0],[0,0,0],[0,0,1]])
        z1=_perceive_with(x,k1)
        z2=_perceive_with(x,k2)
        z3=_perceive_with(x,k3)
        z4=_perceive_with(x,k4)
        z5=_perceive_with(x,k5)
        z6=_perceive_with(x,k6)
        z7=_perceive_with(x,k7)
        z8=_perceive_with(x,k8)
        y = torch.cat((x,z1,z2,z3,z4,z5,z6,z7,z8),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)
        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc00(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)
        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        out = self.fc2(x.mean([1,2]))
        out = F.relu(out)
        out =self.fc3(out)
        return out,x

class SigNCA(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(SigNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,11)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            #x2 = F.relu(x2)
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        x=F.relu(x)
        out = self.fc2(x.mean([1,2]))
        out = F.relu(out)
        out =self.fc3(out)
        
        return out,x


class BasicNCA(nn.Module):
    #Basic implementation of an NCA using a sobel x and y filter for the perception
    
    def __init__(self, channel_n=16, fire_rate=0.5, device="cpu", hidden_size=128, input_channels=3, init_method="standard"):
        super(BasicNCA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        self.fc2 = nn.Linear(channel_n,128)
        self.fc3 = nn.Linear(128,13)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        #Perceptive function, combines 2 sobel x and y outputs with the identity of the cell
    
        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T

        y1 = _perceive_with(x, dx)
        y2 = _perceive_with(x, dy)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1,3)
        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        x = x+dx.transpose(1,3)
        x = x.transpose(1,3)
        return x

    def forward(self, x, steps=32, fire_rate=0.5):
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        out = self.fc2(x.mean([1,2]))
        out = F.relu(out)
        out =self.fc3(out)
        
        return out,x
    