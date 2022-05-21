import model
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import SupportingFunctions as sf
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print('Test code has been started.')

### HYPERPARAMETERS
params = dict([('num_epoch', 100),
               ('batch_size', 1),
               ('learning_rate', 1e-3),
               ('num_workers', 0),          # It should be 0 for Windows machines
               ('exp_num', 7),              # CHANGE EVERYTIME
               ('save_flag', False),
               ('use_cpu', False),
               ('acc_rate', 4),
               ('K', 10)])   

### PATHS          
test_data_path  = 'Knee_Coronal_PD_RawData_392Slices_Test.h5'
test_coil_path  = 'Knee_Coronal_PD_CoilMaps_392Slices_Test.h5'
                   
# 0) Fix randomness for reproducible experiment
torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# 1) Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and (not(params['use_cpu']))) else 'cpu')

# 2) Load Data
dataset = sf.KneeDataset(test_data_path, test_coil_path, params['acc_rate'], num_slice=10)
loaders, datasets= sf.prepare_test_loaders(dataset,params)
mask = dataset.mask.to(device)

####################################################
############## TEST CODE ###########################
####################################################
denoiser = model.ResNet().to(device)
denoiser.load_state_dict(torch.load('model_t__ResNet_100.pt'))
denoiser.eval()
for i, (x0, xref, sens_map, index) in enumerate(loaders['test_loader']):
    with torch.no_grad():
        x0 = x0.to(device)
        xref = xref.to(device)
        sens_map = sens_map.to(device)
        # Forward pass
        xk = x0
        for k in range(params['K']):
            L, zk = denoiser(xk)
            xk = model.DC_layer(x0,zk,L,sens_map,mask)
        
        
        figure = plt.figure()
        plt.imshow(np.abs(x0.cpu().detach().numpy()[0,:,:]),cmap='gray')
        plt.title(f'zero_filled_slice:{index.item():03d}')
        plt.axis('off')
        #plt.show()
        figure.savefig('x0'+f'_fn_{i:03d}'+'.png')   
        
        figure = plt.figure()
        plt.imshow(np.abs(xk.cpu().detach().numpy()[0,:,:]),cmap='gray')
        plt.title(f'MoDL_slice:{index.item():03d}')
        plt.axis('off')
        #plt.show()
        figure.savefig('xk'+f'_fn_{i:03d}'+'.png')  
        
        figure = plt.figure()
        plt.imshow(np.abs(xref.cpu().detach().numpy()[0,:,:]),cmap='gray')
        plt.title(f'reference_slice:{index.item():03d}')
        plt.axis('off')
        #plt.show()
        figure.savefig('xref'+f'_fn_{i:03d}'+'.png')  
