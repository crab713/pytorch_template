import math
from traceback import print_tb
from typing import Tuple
from matplotlib.pyplot import cla
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, init


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3,3), bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.reset_weight()

    def forward(self, inputs : Tensor, state : Tuple[Tensor, Tensor] = None) -> Tuple[Tensor, Tensor]:
        # inputs = [batch, channel, height, width]
        if state is None:
            h_cur, c_cur = self.init_hidden(inputs.size(0), (inputs.size(2), inputs.size(3)))
        else:
            h_cur, c_cur = state

        combined = torch.cat([inputs, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # h,c = [batch, hidden_dim, height, width]
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

    def reset_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class ConvLSTM(nn.Module):
    r"""A LSTM model
    
    Args:
        input_size:
        hidden_size:
        num_layers:
        dropout: if non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer. Default: 0
        bidirectional: if True, becomes a bidirectional LSTM. Default: False
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: int = 0,
                 bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first

        self.cells = []
        for i in range(self.num_directions):
            self.cells.append([])
        for i in range(self.num_directions):
            for j in range(num_layers):
                if j == 0:
                    cell = ConvLSTMCell(input_size, hidden_size)
                else:
                    cell = ConvLSTMCell(hidden_size, hidden_size)
                self.cells[i].append(cell)
                setattr(self, 'cell{}_{}'.format(i,j), cell)

    def check_forward_inputs(self, inputs:Tensor) -> None:
        if len(inputs.shape) != 5:
            raise RuntimeError(
                'Input has inconsistent dimension: got {}, expect: 5'.format(
                    len(inputs.shape)))

        if inputs.size(2) != self.input_size:
            raise RuntimeError(
                'Input has inconsistent input_size: got {}, expect: {}'.format(
                    inputs.size(2), self.input_size))
    
    def each_direction_forward(self, inputs:Tensor, direction: int = 0) -> Tensor:
        r"""
        Inputs: inputs, forward
            - inputs shape:[seq_len, batch, input_size]
            - direction: which direction now, if forward then i mod 2 == 0. Default:0
        
        Outputs: outputs
            - outputs shape:[seq_len, batch, hidden_size]
        """
        seq_len = inputs.size(0)
        step_direction = 1 if direction%2==0 else -1
        
        outputs = []
        h_history = [[] for _ in range(self.num_layers)]
        c_history = [[] for _ in range(self.num_layers)]
        for step in range(seq_len):
            t = step if step_direction==1 else seq_len-step-1
            for cell_idx, cell in enumerate(self.cells[direction]):
                input = inputs[t] if cell_idx==0 else h_history[cell_idx-1][step]
                state = (h_history[cell_idx][step-1], c_history[cell_idx][step-1]) if step!=0 else None

                hn,cn = cell(input, state)
                
                h_history[cell_idx].append(hn)
                c_history[cell_idx].append(cn)
            outputs.append(hn)
        outputs = torch.stack(outputs[::step_direction], dim=0)
        return outputs

    def forward(self, inputs: Tensor) -> Tensor:
        r"""
        Inputs: inputs, (h_n, c_n)
            - inputs shape:[seq_len, batch, input_size, width, height]
            - h_n shape:[num_layer * num_directions, batch, hidden_size, width, height]
            - c_n shape:[num_layer * num_directions, batch, hidden_size, width, height]
        
        Outputs: outputs, (h_n, c_n)
            - outputs shape:[seq_len, batch, num_directions * hidden_size, width, height]
            - h_n shape:[num_layer * num_directions, batch, hidden_size, width, height]
            - c_n shape:[num_layer * num_directions, batch, hidden_size, width, height]
        """
        self.check_forward_inputs(inputs)
        if self.batch_first:
            inputs = inputs.permute(1, 0, 2, 3, 4)
            # NOTE inputs: [seq_len, batch, input_size]

        outputs = []
        for direction in range(self.num_directions):
            output = self.each_direction_forward(inputs, direction)       
            outputs.append(output)
        # exit()
        outputs = torch.cat(outputs, dim=2)
        if self.batch_first:
            outputs = outputs.permute(1, 0, 2, 3, 4)
        return outputs


if __name__ == '__main__':
    lstm = ConvLSTM(1, 10, batch_first=True)
    x = torch.randn(32, 5, 1, 32, 32)
    outputs = lstm(x)
    print(outputs.shape) # torch.Size([32, 5, 10, 32, 32])