import model
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import SupportingFunctions as sf

print('Training code has been started.')

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
train_data_path  = 'Knee_Coronal_PD_RawData_300Slices_Train.h5'
train_coil_path  = 'Knee_Coronal_PD_CoilMaps_300Slices_Train.h5'
                 
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
dataset = sf.KneeDataset(train_data_path,train_coil_path, params['acc_rate'], num_slice=300)
loaders, datasets= sf.prepare_train_loaders(dataset,params,g)
mask = dataset.mask.to(device)

# 3) Create Model structure
denoiser = model.Dw().to(device)
optimizer = torch.optim.Adam(denoiser.parameters(),lr=params['learning_rate'])

loss_arr       = np.zeros(params['num_epoch'])
loss_arr_valid = np.zeros(params['num_epoch'])

for epoch in range(params['num_epoch']):
    print ('-----------------------------')
    print (f'Epoch [{epoch}/{params["num_epoch"]}], \
           Loss training: {loss_arr[epoch-1]:.2f}, \
           Loss validation: {loss_arr_valid[epoch-1]:.2f}')
    print ('-----------------------------')
    for i, (x0, xref, sens_map, index) in enumerate(loaders['train_loader']):
        x0       = x0.to(device)
        xref     = xref.to(device)
        sens_map = sens_map.to(device)
        # Forward pass
        xk = x0
        for k in range(params['K']):
            L, zk = denoiser(xk)
            xk = model.DC_layer(x0,zk,L,sens_map,mask)
        
        optimizer.zero_grad()
        # Loss calculation
        loss  = torch.sum(torch.real(xref - xk)**2 + (torch.imag(xref - xk)**2))
        loss.requres_grad = True
        loss_arr[epoch]  += loss.item()/len(datasets['train_dataset'])
        loss.backward()
        
        # Optimize
        optimizer.step()
        if ((epoch+1)%20==0):
          torch.save(denoiser.state_dict(), 'model_t_' + f'_fn0_{epoch+1:03d}'+ '.pt')

    for i, (x0, xref, sens_map, index) in enumerate(loaders['valid_loader']):
        with torch.no_grad():
            x0 = x0.to(device)
            xref = xref.to(device)
            sens_map = sens_map.to(device)
            # Forward pass
            xk = x0
            for k in range(params['K']):
                L, zk = denoiser(xk)
                xk = model.DC_layer(x0,zk,L,sens_map,mask)
            loss  = (torch.sum(torch.real(xref - xk)**2 + (torch.imag(xref - xk)**2)))
            loss_arr_valid[epoch] += loss.item()/len(datasets['valid_dataset'])

figure = plt.figure()
n = np.arange(1,params['num_epoch']+1)
plt.plot(n,loss_arr,n,loss_arr_valid)
figure.savefig('loss_graph.png')