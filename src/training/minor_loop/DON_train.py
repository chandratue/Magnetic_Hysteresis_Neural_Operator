###########################################################################
############# Importing Libraries and paths ###############################
###########################################################################
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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
from models.DON import DNN, train_step, test_step

# Loss
loss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/logs/DON/minor_loop/'))
os.makedirs(loss_dir, exist_ok=True)
train_loss_path = os.path.join(loss_dir, 'train_loss_list.pkl')
test_loss_path = os.path.join(loss_dir, 'test_loss_list.pkl')

# Model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/trained_models/DON/minor_loop/'))
model_save_path = os.path.join(model_dir, 'trained_model.pth')

############ Seed #############
np.random.seed(1234)

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
x_test = np.linspace(0, 1, 198).reshape(-1,1)

############################################################################
################# Hyperparameters and Initialize model #####################
############################################################################

# number of points on which input function is sampled
points = 198

# number of basis functions and coefficients
G_dim = 100

# Branch Net
layers_f = [points] + [200] * 8 + [G_dim]

# Problem dimension
x_dim = 1

# Trunk Net
layers_x = [x_dim] + [200] * 8 + [G_dim]

model = DNN()

W_branch, b_branch = model.hyper_initial(layers_f)
W_trunk, b_trunk = model.hyper_initial(layers_x)

n = 0
nmax = 100000
lr = 5e-5

optimizer = optim.Adam(list(W_branch) + list(b_branch) + list(W_trunk) + list(b_trunk), lr=lr)

train_loss_list = []
test_loss_list = []

############################################################################
########################### Training model #################################
############################################################################

while n <= nmax:
    x_train_tensor = torch.from_numpy(x_train).float()
    H_train_scaled_tensor = torch.from_numpy(H_train_scaled).float()
    B_train_tensor = torch.from_numpy(B_train).float()

    loss_train, B_train_pred = train_step(model, W_branch, b_branch, W_trunk,
                                          b_trunk, H_train_scaled_tensor, x_train_tensor, B_train_tensor,
                                          optimizer)

    B_train_pred_np = B_train_pred.detach().numpy()

    loss_test, B_test_pred = test_step(model, W_branch, b_branch, W_trunk, b_trunk, H_test_scaled, x_test, B_test,
                                       optimizer)

    if n % 10 == 0:
        print(f"Iteration: {n} Train_loss:{loss_train}, Test_loss: {loss_test}")
    train_loss_list.append(loss_train)
    test_loss_list.append(loss_test)
    n = n + 1

##### Saving loss history ########

with open(train_loss_path, 'wb') as f:
    pickle.dump(train_loss_list, f)

with open(test_loss_path, 'wb') as f:
    pickle.dump(test_loss_list, f)

####### Saving trained model #######

# After the training loop, add the following lines to save the model
torch.save({
    'W_branch': W_branch,
    'b_branch': b_branch,
    'W_trunk': W_trunk,
    'b_trunk': b_trunk,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, model_save_path)
print("Model Saved Successfully")
