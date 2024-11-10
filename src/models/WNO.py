import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

# Set the seed for NumPy
np.random.seed(1234)

# Set the seed for PyTorch
torch.manual_seed(1234)

""" Load required packages 

It requires the packages
-- "Pytorch Wavelets"
    see https://pytorch-wavelets.readthedocs.io/en/latest/readme.html
    ($ git clone https://github.com/fbcotter/pytorch_wavelets
     $ cd pytorch_wavelets
     $ pip install .)

-- "PyWavelets"
    https://pywavelets.readthedocs.io/en/latest/install.html
    ($ conda install pywavelets)

-- "Pytorch Wavelet Toolbox"
    see https://github.com/v0lta/PyTorch-Wavelet-Toolbox
    ($ pip install ptwt)
"""



try:
    import ptwt, pywt
    from ptwt.conv_transform_3 import wavedec3, waverec3
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    from pytorch_wavelets import DWT, IDWT
except ImportError:
    print('Wavelet convolution requires <Pytorch Wavelets>, <PyWavelets>, <Pytorch Wavelet Toolbox> \n \
                    For Pytorch Wavelet Toolbox: $ pip install ptwt \n \
                    For PyWavelets: $ conda install pywavelets \n \
                    For Pytorch Wavelets: $ git clone https://github.com/fbcotter/pytorch_wavelets \n \
                                          $ cd pytorch_wavelets \n \
                                          $ pip install .')


""" Def: 1d Wavelet convolutional layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='symmetric'):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if np.isscalar(size):
            self.size = size
        else:
            raise Exception("size: WaveConv1d accepts signal length in scalar only")
        self.wavelet = wavelet
        self.mode = mode
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode)
        dummy_data = torch.randn( 1,1,self.size )
        mode_data, _ = self.dwt_(dummy_data)
        self.modes1 = mode_data.shape[-1]

        # Parameter initilization
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # Convolution
    def mul1d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x )
                  1D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Input parameters:
        -----------------
        x : tensor, shape-[Batch * Channel * x]

        Output parameters:
        ------------------
        x : tensor, shape-[Batch * Channel * x]
        """
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT1D(wave=self.wavelet, J=self.level+factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)

        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT1D(wave=self.wavelet, J=self.level-factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)

        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)

        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]

        # Multiply the final low pass wavelet coefficients
        out_ft = self.mul1d(x_ft, self.weights1)
        # Multiply the final high pass wavelet coefficients
        out_coeff[-1] = self.mul1d(x_coeff[-1].clone(), self.weights2)

        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(x.device)
        x = idwt((out_ft, out_coeff))
        return x


# %%
""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x) = g(K.v + W.v)(x).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: scalar (for 1D), right support of 1D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range
        self.padding = padding

        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 2: (a(x), x)
        for i in range( self.layers ):
            self.conv.append( WaveConv1d(self.width, self.width, self.level, self.size, self.wavelet) )
            self.w.append( nn.Conv1d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)              # Shape: Batch * x * Channel
        x = x.permute(0, 2, 1)       # Shape: Batch * Channel * x
        if self.padding != 0:
            x = F.pad(x, [0,self.padding])

        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x)
            if index != self.layers - 1:   # Final layer has no activation
                x = F.mish(x)        # Shape: Batch * Channel * x

        if self.padding != 0:
            x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)       # Shape: Batch * x * Channel
        x = F.gelu( self.fc1(x) )    # Shape: Batch * x * Channel
        x = self.fc2(x)              # Shape: Batch * x * Channel
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
