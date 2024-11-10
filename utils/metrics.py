import torch

def relative_error(u_pred, u_exact):
    # Ensure the inputs are torch tensors
    u_pred = torch.tensor(u_pred, dtype=torch.float32)
    u_exact = torch.tensor(u_exact, dtype=torch.float32)

    # Compute the mean squared error
    mse = torch.mean((u_pred - u_exact)**2)

    # Compute the mean of the exact values squared
    mse_exact = torch.mean(u_exact**2)

    # Compute the relative error
    relative_error = torch.sqrt(mse / mse_exact)

    return relative_error

# Example usage
# u_test_pred = torch.tensor([your_predicted_values])
# u_exact = torch.tensor([your_exact_values])
# error = relative_error_test(u_test_pred, u_exact)
# print(f"Relative Error: {error.item()}")
