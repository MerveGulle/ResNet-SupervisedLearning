import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader

### DEFINE FFT2 AND IFFT2 FUNCTIONS
# y = FFT(x): FFT of one slice image to kspace: [1 Nx Ny Nc] --> [1 Nx Ny Nc]
def fft2(img):
    _, Nx, Ny, Nc = img.shape
    # fft = torch.zeros((1, Nx, Ny, Nc), dtype=torch.cfloat)
    fft = img
    for coil in np.arange(Nc):
        A               = torch.squeeze(img[0,:,:,coil])
        A               = torch.fft.ifftshift(A)
        A               = torch.fft.fft2(A, norm='ortho')
        fft[0,:,:,coil] = torch.fft.fftshift(A)

    return fft

# x = iFFT(y): iFFT of one slice kspace to image: [1 Nx Ny Nc] --> [1 Nx Ny Nc]
def ifft2(kspace):
    _, Nx, Ny, Nc = kspace.shape
    # ifft = torch.zeros((1, Nx, Ny, Nc), dtype=torch.cfloat)
    ifft = kspace
    
    for coil in np.arange(Nc):
        A                  = torch.squeeze(kspace[0,:,:,coil])
        A                  = torch.fft.ifftshift(A)
        A                  = torch.fft.ifft2(A, norm='ortho')
        ifft[0,:,:,coil]   = torch.fft.fftshift(A)

    return ifft

# y = Ex: encoding one slice image to kspace: [1 Nx Ny] --> [1 Nx Ny Nc]
# S: sensitivity map
def encode(x,S,mask):
    y = S*x[:,:,:,None]       # sensitivity map element-wise multiplication
    y = fft2(y)             # Fourier transform
    y = y*mask[None,:,:,None] # undersampling
    return y

# y = E'x: reconstruction from kspace to image space: [1 Nx Ny Nc] --> [1 Nx Ny]
# S: sensitivity map
def decode(x,S):
    y = ifft2(x)               # Inverse fourier transform
    y = y*torch.conj(S)
    y = y.sum(axis=3)
    return y 

# Normalised Mean Square Error (NMSE)
# gives the nmse between x and xref
def nmse(x,xref):
    out1 = torch.sum(torch.real(x-xref)**2 + (torch.imag(x-xref)**2))
    out2 = torch.sum(torch.real(xref-torch.mean(xref))**2 + (torch.imag(xref-torch.mean(xref))**2))
    return out1/out2

# Structural Similarity Index (SSIM)
# gives similarity index of x and y images 
def ssim(x,y):
    _,Nx,Ny = x.shape 
    N = Nx*Ny
    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    S_xx = (torch.sum((torch.real(x-mu_x))**2) + torch.sum((torch.imag(x-mu_x))**2))/(N-1)
    S_yy = (torch.sum((torch.real(y-mu_y))**2) + torch.sum((torch.imag(y-mu_y))**2))/(N-1)
    S_xy = (torch.sum(torch.real(x-mu_x)*torch.real(y-mu_y)) + torch.sum(torch.imag(x-mu_x)*torch.imag(y-mu_y)))/(N-1)
    out = ((2*mu_x*mu_y)*(2*S_xy))/((mu_x**2+mu_y**2)*(S_xx+S_yy))
    return torch.abs(out)

class KneeDataset():
    def __init__(self,data_path,coil_path,R,num_slice,num_ACS=24):
        f = h5py.File(data_path, "r")
        start_slice = 10
        r = 30
        self.kspace    = f['kspace'][start_slice:start_slice+num_slice*r:r]
        self.kspace    = torch.from_numpy(self.kspace)
        
        self.n_slices  = self.kspace.shape[0]
        
        S = h5py.File(coil_path, "r")
        _, value = list(S.items())[0]
        self.sens_map    = value[start_slice:start_slice+num_slice*r:r]
        self.sens_map    = torch.from_numpy(self.sens_map)
        
        self.mask = torch.zeros((self.kspace.shape[1],self.kspace.shape[2]), dtype=torch.cfloat)
        self.mask[:,::R] = 1.0
        self.mask[:,(self.kspace.shape[2]-num_ACS)//2:(self.kspace.shape[2]+num_ACS)//2] = 1.0
        
        self.x0   = torch.empty(self.kspace.shape[0:3], dtype=torch.cfloat)
        self.xref = torch.empty(self.kspace.shape[0:3], dtype=torch.cfloat)
        self.R    = 1/(torch.abs(self.mask).sum()/(self.kspace.shape[1]*self.kspace.shape[2]))
        for i in range(self.kspace.shape[0]):
            self.x0[i] = decode(self.kspace[i:i+1]*self.mask[None,:,:,None],self.sens_map[i:i+1])
            norm_value = torch.max(torch.abs(self.x0[i:i+1]))
            self.x0[i] = self.x0[i:i+1] / norm_value
            
            self.xref[i] = decode(self.kspace[i:i+1],self.sens_map[i:i+1]) / norm_value
     
    def __getitem__(self,index):
        return self.x0[index], self.xref[index], self.sens_map[index], index
    def __len__(self):
        return self.n_slices   

# complex 1 channel to real 2 channels
def ch1to2(data1):       
    return torch.cat((data1.real,data1.imag),0)
# real 2 channels to complex 1 channel
def ch2to1(data2):       
    return data2[0:1,:,:] + 1j * data2[1:2,:,:] 

def prepare_train_loaders(dataset,params,g):
    train_num  = int(dataset.n_slices * 0.8)
    valid_num  = dataset.n_slices - train_num

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_num,valid_num],  generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(dataset       = train_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)

    valid_loader = DataLoader(dataset       = valid_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)

    full_loader= DataLoader(dataset         = dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = False,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)
    
    datasets = dict([('train_dataset', train_dataset),
                     ('valid_dataset', valid_dataset)])  
    
    loaders = dict([('train_loader', train_loader),
                    ('valid_loader', valid_loader),
                    ('full_loader', full_loader)])

    return loaders, datasets

def prepare_test_loaders(test_dataset,params):
    test_loader  = DataLoader(dataset       = test_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'])
    
    datasets = dict([('test_dataset', test_dataset)])  
    
    loaders = dict([('test_loader', test_loader)])

    return loaders, datasets

# Normalised L1-L2 loss calculation
# loss = normalised L1 loss + normalised L2 loss
def L1L2Loss(ref, out):
    N = ref.size
    diff = ref - out    
    L1 = ((torch.sum(torch.real(diff))**2 + torch.sum(torch.real(diff)**2)) 
          /(torch.sum(torch.real(ref))**2 + torch.sum(torch.real(ref)**2)))
    L2 = torch.sum(torch.abs(diff)) / torch.sum(torch.abs(ref))
    return (L1 + L2) / N








