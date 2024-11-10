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
from models.RNN import RNN

# Loss
loss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/logs/RNN/minor_loop/'))
os.makedirs(loss_dir, exist_ok=True)
train_loss_path = os.path.join(loss_dir, 'train_loss_list.pkl')
test_loss_path = os.path.join(loss_dir, 'test_loss_list.pkl')

# Model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/trained_models/RNN/minor_loop/'))
model_save_path = os.path.join(model_dir, 'trained_model.pth')

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
num_epochs = 20000

input_data = H_train_scaled.T
target_data = B_train.T

# Convert data to tensors
input_tensor = torch.tensor(input_data).view(batch_size, sequence_length, input_size).float()
target_tensor = torch.tensor(target_data).view(batch_size, sequence_length, output_size).float()

# Create RNN instance
rnn = RNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)

# Initialize a list to store loss values
loss_history_train = []

# Training loop
for epoch in range(num_epochs):
    # Set initial hidden state
    hidden = torch.zeros(1, batch_size, hidden_size)

    # Forward pass
    output, hidden = rnn(input_tensor, hidden)
    loss = criterion(output, target_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history_train.append(loss.item())

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.5e}')

##### Saving loss history ########

with open(train_loss_path, 'wb') as f:
    pickle.dump(loss_history_train, f)

####### Saving trained model #######

# After the training loop, add the following lines to save the model
torch.save({'model_state_dict': rnn.state_dict()}, model_save_path)
print("Model Saved Successfully")
