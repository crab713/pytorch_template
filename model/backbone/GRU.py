from torch.autograd import Variable
import torch
import torch.nn as nn

#input data size:BxFxT

class GRU(nn.Module):
    def __init__(self,c_in,c_out,tem_size):
        super(GRU,self).__init__()
        self.mlp=nn.Linear(c_out+c_in,c_out*4)
        self.c_out=c_out
        self.tem_size=tem_size

    def forward(self,x):
        shape = x.shape
        h = Variable(torch.zeros((shape[0],self.c_out))).cuda()
        c = Variable(torch.zeros((shape[0],self.c_out))).cuda()
        out=[]
        for k in range(self.tem_size):
            input1=x[:,:,k]
            tem1=torch.cat((input1,h),1)
            fea1=self.mlp(tem1)
            i,j,f,o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c=c*torch.sigmoid(f)+torch.sigmoid(i)*torch.tanh(j)
            new_h=torch.tanh(new_c)*(torch.sigmoid(o))
            c=new_c
            h=new_h
            out.append(new_h)
        x=torch.stack(out,-1)
        return x