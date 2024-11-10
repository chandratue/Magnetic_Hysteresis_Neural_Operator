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
from models.GRU import GRU

# Model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/trained_models/GRU/FORC/'))
model_save_path = os.path.join(model_dir, 'trained_model.pth')

# saving pred and errors
preds_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/preds/GRU/FORC/'))
preds_save_path = os.path.join(preds_dir, 'pred.npz')
# Error
err_save_path = os.path.join(preds_dir, 'error.csv')

# Metric
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from utils.metrics import relative_error

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

############ Seed #############
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

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

# Toy problem data
input_size = 1000  # Updated input size
hidden_size = 128
output_size = 1000  # Updated output size
sequence_length = 198
batch_size = 1

input_data = H_test_scaled.T
target_data = B_test.T

# Convert data to tensors
input_tensor = torch.tensor(input_data).view(batch_size, sequence_length, input_size).float()
target_tensor = torch.tensor(target_data).view(batch_size, sequence_length, output_size).float()

# Loss and optimizer
criterion = nn.MSELoss()

#########################################
######### Load Trained Model ############
#########################################
# Create GRU instance
gru = GRU(input_size, hidden_size, output_size)

lr = 1e-4

checkpoint = torch.load(model_save_path)
gru.load_state_dict(checkpoint['model_state_dict'])
print("Saved model loaded successfully.")

with torch.no_grad():
    hidden_pred = torch.zeros(1, batch_size, hidden_size)
    prediction, _ = gru(input_tensor, hidden_pred)

prediction = prediction.squeeze(0).detach()

prediction  = prediction.view(198, 1000)
target_tensor = target_tensor.view(198, 1000)

Error = relative_error(prediction.detach().numpy(), target_tensor.detach().numpy())
Error_np = Error.detach().numpy()
print('Error: ', Error_np)

MAE = mean_absolute_error(prediction.detach().numpy(), target_tensor.detach().numpy())
print('MAE: ', MAE)

MSE = mean_squared_error(prediction.detach().numpy(), target_tensor.detach().numpy())
RMSE = np.sqrt(MSE)
print('RMSE: ', RMSE)

Error_np = np.array([Error_np])
# Save error in a CSV file
np.savetxt(err_save_path, Error_np, delimiter=',')

torch.save({'B_pred': prediction}, preds_save_path)
print("Test data saved successfully")

gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[:, :])

ax1.plot(H_test[20, :], prediction.T.detach().numpy()[20, :], '-k', lw=2.0, label="Predicted")

for i in range(10, 500):
    ax1.plot(H_test[i, :], prediction.T.detach().numpy()[i, :], '-k', lw=2.0)

ax1.plot(H_test[20, :], B_test[20, :], '--r', lw=2.0, label="Actual")

for i in range(10, 500):
    ax1.plot(H_test[i, :], B_test[i, :], '--r', lw=2.0)

ax1.set_xlabel('$H$')
ax1.set_ylabel('$B$')
ax1.legend(frameon=True, loc='best')
plt.show()
