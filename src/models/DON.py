import numpy as np
import torch
import torch.nn as nn

# Set the seed for NumPy
np.random.seed(1234)

# Set the seed for PyTorch
torch.manual_seed(1234)

#############################################################################
########################## DON Model #######################################
############################################################################
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

    # Initialization for DNNs
    def hyper_initial(self, layers):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l - 1]
            out_dim = layers[l]
            std = np.sqrt(2.0 / (in_dim + out_dim))
            weight = nn.Parameter(torch.randn(in_dim, out_dim) * std)
            bias = nn.Parameter(torch.randn(1, out_dim) * std)
            W.append(weight)
            b.append(bias)
        return W, b

    def fnn_B(self, X, W, b):
        A = X
        L = len(W)
        for i in range(L - 1):
            A = torch.tanh(torch.matmul(A, W[i]) + b[i])
        Y = torch.matmul(A, W[-1]) + b[-1]
        return Y

    def fnn_T(self, X, W, b):
        A = X
        L = len(W)
        for i in range(L - 1):
            A = torch.tanh(torch.matmul(A, W[i]) + b[i])
        Y = torch.matmul(A, W[-1]) + b[-1]
        return Y

# x (100, 1)--input trunk
# u (1000, 100)--
# v (1000, 100)--input branch

def train_step(model, W_branch, b_branch, W_trunk, b_trunk, v, x, u, optimizer):
    train_vars = W_branch + b_branch + W_trunk + b_trunk
    optimizer.zero_grad()
    u_out_branch = model.fnn_B(v, W_branch, b_branch)
    u_out_trunk = model.fnn_T(x, W_trunk, b_trunk)
    # u_pred = torch.einsum('ij,jj->ij', u_out_branch, u_out_trunk)
    u_pred = torch.einsum('ij,kj->ik', u_out_branch, u_out_trunk)

    loss = torch.mean(torch.square(u_pred - u))
    loss.backward()
    optimizer.step()
    return loss.item(), u_pred.detach()


def test_step(model, W_branch, b_branch, W_trunk, b_trunk, v, x, u, optimizer):
    x_test_tensor = torch.from_numpy(x).float()
    v_test_tensor = torch.from_numpy(v).float()
    u_test_tensor = torch.from_numpy(u).float()

    with torch.no_grad():
        u_out_branch = model.fnn_B(v_test_tensor, W_branch, b_branch)
        u_out_trunk = model.fnn_T(x_test_tensor, W_trunk, b_trunk)
        # u_pred = torch.einsum('ij,jj->ij', u_out_branch, u_out_trunk)
        u_pred = torch.einsum('ij,kj->ik', u_out_branch, u_out_trunk)

        loss = torch.mean(torch.square(u_pred - u_test_tensor))

    return loss.item(), u_pred.detach().numpy()
