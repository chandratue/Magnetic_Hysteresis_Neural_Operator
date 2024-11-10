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

file_path_B_train = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/minor_loop/B_train.npz'))
file_path_B_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/minor_loop/B_test.npz'))
file_path_H_train = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/minor_loop/H_train.npz'))
file_path_H_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/minor_loop/H_test.npz'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.scaling import min_max_scaling, inverse_min_max_scaling

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.FNO import FNO1d

# Model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/trained_models/FNO/minor_loop/'))
model_save_path = os.path.join(model_dir, 'trained_model.pth')

# saving pred and errors
preds_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/preds/FNO/rate_independent/'))
preds_save_path = os.path.join(preds_dir, 'pred.npz')
# Error
err_save_path = os.path.join(preds_dir, 'error.csv')

# Metric
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from utils.metrics import relative_error



############ Seed #############
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

batch_size = 500
modes = 4    #16                   # number of Fourier modes to multiply, at most floor(N/2) + 1
width = 8    #64                   # number of hidden channel

##########################################
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
x_train = np.linspace(0, 1, 198).reshape(-1,1)
x_test = np.linspace(0, 10, 198).reshape(-1,1)

# load training data


H_train_scaled = torch.Tensor(H_train_scaled)
B_train = torch.Tensor(B_train)

H_test_scaled = torch.Tensor(H_test_scaled)
B_test = torch.Tensor(B_test)

x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)

# concatenate the spatial grid and the spatial solution
Ht_train = torch.cat([H_train_scaled.reshape(1000, -1, 1), x_train.repeat(1000, 1, 1)], dim=2)
Ht_test = torch.cat([H_test_scaled.reshape(1000, -1, 1), x_test.repeat(1000, 1, 1)], dim=2)
print(f'[Dataset] Ht_train: {Ht_train.shape}, B_train: {B_train.shape}')
print(f'[Dataset] Ht_test: {Ht_test.shape}, B_test: {B_test.shape}')

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

#########################################
######### Load Trained Model ############
#########################################
lr = 1e-4
model = FNO1d(modes, width)

checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
print("Saved model loaded successfully.")

# model.eval()
# with torch.no_grad():
for x, y in test_loader:
    x, y = x, y
    B_test_pred = model(x)

B_test_pred = B_test_pred.squeeze()

Error = relative_error(B_test_pred, y)
Error_np = Error.detach().numpy()
print('Error: ', Error_np)

Error_np = np.array([Error_np])
# Save error in a CSV file
np.savetxt(err_save_path, Error_np, delimiter=',')

torch.save({'B_pred': B_test_pred}, preds_save_path)
print("Test data saved successfully")

#v_test = inverse_min_max_scaling(v_test, H_data_original, -1, 1)

gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[:, :])

ax1.plot(H_test[20, :], B_test_pred.detach().numpy()[20, :], '-k', lw=2.0, label="Predicted")

for i in range(10, 500):
    ax1.plot(H_test[i, :], B_test_pred.detach().numpy()[i, :], '-k', lw=2.0)

ax1.plot(H_test[20, :], B_test[20, :], '--r', lw=2.0, label="Actual")

for i in range(10, 500):
    ax1.plot(H_test[i, :], B_test[i, :], '--r', lw=2.0)

ax1.set_xlabel('$H$')
ax1.set_ylabel('$B$')
ax1.legend(frameon=True, loc='best')
plt.show()