import numpy as np
import torch
import torch.nn as nn
import SupportingFunctions as sf

# define RB:residual block (conv + ReLU + conv + xScale)
class RB(nn.Module):
    def __init__(self, C=0.1):
        super().__init__()
        self.RBconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.RBconv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.C     = C
    def forward(self, x):
        y = self.RBconv1(x)
        y = self.relu(y)
        y = self.RBconv2(y)
        y = y*self.C
        y = y + x
        return y

# x0  : initial solution
# zn  : Output of nth denoiser block
# L   : regularization coefficient
# tol : tolerance for breaking the CG iteration
def DC_layer(x0,zn,L,S,mask,tol=0,cg_iter=10):
    _,Nx,Ny = x0.shape
    # xn = torch.zeros((Nx, Ny), dtype=torch.cfloat)
    xn = x0[0,:,:]*0
    a  = torch.squeeze(x0 + L*zn)
    p  = a
    r  = a
    for i in np.arange(cg_iter):
        delta = torch.sum(r*torch.conj(r)).real/torch.sum(a*torch.conj(a)).real
        if(delta<tol):
            break
        else:
            p1 = p[None,:,:]
            q  = torch.squeeze(sf.decode(sf.encode(p1,S,mask),S)) + L* p
            t  = (torch.sum(r*torch.conj(r))/torch.sum(q*torch.conj(p)))
            xn = xn + t*p 
            rn = r  - t*q 
            p  = rn + (torch.sum(rn*torch.conj(rn))/torch.sum(r*torch.conj(r)))*p
            r  = rn
            
    return xn[None,:,:]

# define ResNet Block
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.RB1   = RB()
        self.RB2   = RB()
        self.RB3   = RB()
        self.RB4   = RB()
        self.RB5   = RB()
        self.RB6   = RB()
        self.RB7   = RB()
        self.RB8   = RB()
        self.RB9   = RB()
        self.RB10  = RB()
        self.RB11  = RB()
        self.RB12  = RB()
        self.RB13  = RB()
        self.RB14  = RB()
        self.RB1  = RB()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.L = nn.Parameter(torch.tensor(0.05, requires_grad=True))
    def forward(self, x):
        z = sf.ch1to2(x)[None,:,:,:].float()
        z = self.conv1(z)
        r = self.conv2(z)
        r = self.RB1(r)
        r = self.RB2(r)
        r = self.RB3(r)
        r = self.RB4(r)
        r = self.RB5(r)
        r = self.RB6(r)
        r = self.RB7(r)
        r = self.RB8(r)
        r = self.RB9(r)
        r = self.RB10(r)
        r = self.RB11(r)
        r = self.RB12(r)
        r = self.RB13(r)
        r = self.RB14(r)
        r = self.RB15(r)
        r = self.conv3(r)
        z = r + z
        z = self.conv4(z)
        z = sf.ch2to1(z[0,:,:,:])
        return self.L, z