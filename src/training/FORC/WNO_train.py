#############################################################################
############# Importing Libraries and paths ###############################
############################################################################
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.io import loadmat
import matplotlib.gridspec as gridspec
import sys
import os
import pickle

file_path_B_train = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/FORC/B_train.npz'))
file_path_B_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/FORC/B_test.npz'))
file_path_H_train = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/FORC/H_train.npz'))
file_path_H_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/FORC/H_test.npz'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.scaling import min_max_scaling, inverse_min_max_scaling

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.WNO import WNO1d

# Loss
loss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/logs/WNO/FORC/'))
os.makedirs(loss_dir, exist_ok=True)
train_loss_path = os.path.join(loss_dir, 'train_loss_list.pkl')
test_loss_path = os.path.join(loss_dir, 'test_loss_list.pkl')

# Model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/trained_models/WNO/FORC/'))
model_save_path = os.path.join(model_dir, 'trained_model.pth')

############ Seed #############
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

#########################################
############# Loading Data ##############
#########################################

B_train_load = np.load(file_path_B_train)
B_test_load = np.load(file_path_B_test)
H_train_load = np.load(file_path_H_train)
H_test_load = np.load(file_path_H_test)

B_train = B_train_load['B_train']
B_test = B_test_load['B_test']
H_train = H_train_load['H_train']
H_test = H_test_load['H_test']

####### Scaling data ###########
H_train_scaled = min_max_scaling(H_train)
H_test_scaled = min_max_scaling(H_test)

###### Output domain for Branch net #########
x_train = np.linspace(0, 1, 198).astype(np.float64).reshape(-1,1)
x_test = np.linspace(0, 1, 198).astype(np.float64).reshape(-1,1)

#========================#
# Training parameters
#========================#
batch_size = 100
learning_rate = 1e-3

num_epoch = 10000
step_size = 20   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 8        # lavel of wavelet decomposition
width = 64       # uplifting dimension
layers = 4       # no of wavelet layers

h = 198
grid_range = 1
in_channel = 2   # (a(x), x) for this case

#========================#
# dataset information
#========================#
# load training data


H_train_scaled = torch.Tensor(H_train_scaled)
B_train = torch.Tensor(B_train)

H_test_scaled = torch.Tensor(H_test_scaled)
B_test = torch.Tensor(B_test)

x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)

# concatenate the spatial grid and the spatial solution
# Ht_train = torch.cat([H_train_scaled.reshape(1000, -1, 1), x_train.repeat(1000, 1, 1)], dim=2)
# Ht_test = torch.cat([H_test_scaled.reshape(1000, -1, 1), x_test.repeat(1000, 1, 1)], dim=2)
# print(f'[Dataset] Ht_train: {Ht_train.shape}, B_train: {B_train.shape}')
# print(f'[Dataset] Ht_test: {Ht_test.shape}, B_test: {B_test.shape}')

Ht_train = H_train_scaled.unsqueeze(2)
Ht_test = H_test_scaled.unsqueeze(2)

print("H train shape: ", Ht_train.shape)
print("H test shape: ", Ht_test.shape)

# create data loader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Ht_train, B_train),
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Ht_test, B_test),
    batch_size=batch_size,
    shuffle=False
)

# define a model
model = WNO1d(width=width, level=level, layers=layers, size=h, wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


#loss_func = LpLoss(size_average=False) # used in original FNO paper
loss_func = nn.MSELoss() # MSE loss

loss_history_train, loss_history_test = [], []
# start training
for epoch in range(num_epoch):
    model.train()
    for x, y in train_loader:
        x, y = x, y
        optimizer.zero_grad()
        out = model(x)

        loss_train = loss_func(out.view(batch_size, -1), y.view(batch_size, -1))
        loss_train.backward()

        optimizer.step()
        loss_history_train.append(loss_train.item())

    #scheduler.step()  # change the learning rate
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x, y
            out = model(x)
            loss_test = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    loss_history_test.append(loss_test)

    if epoch % 10 == 0:
        print(f"Iteration: {epoch} Train_loss:{loss_history_train[-1]:.2e}, Test_loss: {loss_history_test[-1]:.2e}")


print('Training Finished')

##### Saving loss history ########

with open(train_loss_path, 'wb') as f:
    pickle.dump(loss_history_train, f)

with open(test_loss_path, 'wb') as f:
    pickle.dump(loss_history_test, f)

####### Saving trained model #######

# After the training loop, add the following lines to save the model
torch.save({'model_state_dict': model.state_dict()}, model_save_path)
print("Model Saved Successfully")

