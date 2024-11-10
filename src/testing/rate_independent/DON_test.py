#############################################################################
############# Importing Libraries and paths ###############################
############################################################################
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

# Define the path to the saved model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/trained_models/DON/minor_loop/'))
model_save_path = os.path.join(model_dir, 'trained_model.pth')

# Model
preds_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/preds/DON/minor_loop/'))
preds_save_path = os.path.join(preds_dir, 'pred.npz')
# Error
err_save_path = os.path.join(preds_dir, 'error.csv')

# Metric
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from utils.metrics import relative_error

############ Seed #############
np.random.seed(1234)

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
x_train = np.linspace(0, 1, 198).reshape(-1,1)
x_test = np.linspace(0, 10, 198).reshape(-1,1)

#########################################
######### Load Trained Model ############
#########################################
lr = 5e-5
model = DNN()

checkpoint = torch.load(model_save_path)
W_branch = checkpoint['W_branch']
b_branch = checkpoint['b_branch']
W_trunk = checkpoint['W_trunk']
b_trunk = checkpoint['b_trunk']
optimizer = optim.Adam(list(W_branch) + list(b_branch) + list(W_trunk) + list(b_trunk), lr=lr)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("Saved model loaded successfully.")

_, B_test_pred = test_step(model, W_branch, b_branch, W_trunk, b_trunk, H_test_scaled, x_test, B_test,
                                   optimizer)

Error = relative_error(B_test_pred, B_test)
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

ax1.plot(H_test[20, :], B_test_pred[20, :], '-k', lw=2.0, label="Predicted")

for i in range(10, 500):
    ax1.plot(H_test[i, :], B_test_pred[i, :], '-k', lw=2.0)

ax1.plot(H_test[20, :], B_test[20, :], '--r', lw=2.0, label="Actual")

for i in range(10, 500):
    ax1.plot(H_test[i, :], B_test[i, :], '--r', lw=2.0)

ax1.set_xlabel('$H$')
ax1.set_ylabel('$B$')
ax1.legend(frameon=True, loc='best')
plt.show()