import numpy as np
import torch
import torch.nn as nn
import SupportingFunctions as sf

# define convolution block (conv + BN + ReLU)
def conv_block(in_channels, out_channels, relu=False):
    layers = [
        #define 2D Convolutions
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        #define batch normalization
        #nn.BatchNorm2d(out_channels)
        ]
    if relu: #add activation
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

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

# define MoDL based algorithm
class Dw(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(2,  64, relu=True)
        self.conv2 = conv_block(64, 64, relu=True)
        self.conv3 = conv_block(64, 64, relu=True)
        self.conv4 = conv_block(64, 64, relu=True)
        self.conv5 = conv_block(64, 2,  relu=False)
        self.L = nn.Parameter(torch.tensor(0.05, requires_grad=True))
    def forward(self, x):
        z = sf.ch1to2(x)[None,:,:,:].float()
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.conv4(z)
        z = self.conv5(z)
        z = sf.ch2to1(z[0,:,:,:])
        z = z + x
        return self.L, z