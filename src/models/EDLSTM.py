import torch
import torch.nn as nn
import numpy as np

# Set the seed for NumPy
np.random.seed(1234)

# Set the seed for PyTorch
torch.manual_seed(1234)

class EDLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EDLSTM, self).__init__()

        self.hidden_size = hidden_size

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(output_size, hidden_size, batch_first=True)

        # Final output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, decoder_input, encoder_hidden):
        # Encoder: process the input sequence
        encoder_output, encoder_hidden = self.encoder_lstm(encoder_input, encoder_hidden)

        # Decoder: use the encoder's hidden state as initial hidden state for decoder
        decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, encoder_hidden)

        # Apply linear layer to decoder's output
        output = self.fc(decoder_output)

        return output, decoder_hidden

    def init_hidden(self, batch_size):
        # Initialize the hidden and cell states with zeros
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

