import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class my_lstm(nn.Module):
    def __init__(self):
        super(my_lstm,self).__init__()
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(32*28*28,10)


    def forward(self,input):
        batch_size,row,col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()

        x = torch.cat((input, h), 1)
        f = self.conv_f(x)
        i = self.conv_i(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = c*f+i*g
        h = o*F.tanh(c)
        
        #out = h[-1,:,:]
        out = h.view(h.size(0),32*28*28) 
        #64 25088 
        out = self.classifier(out)       
        return out