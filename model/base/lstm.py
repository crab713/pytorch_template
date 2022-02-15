import math
from traceback import print_tb
from typing import Tuple
from matplotlib.pyplot import cla
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, init

class LSTMCell(nn.Module):
    r"""A LSTM cell
    
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
    Inputs: input, (h_0, c_0)
    
    Outputs: (h_1, c_1)
    """
    __constants__ = ['input_size', 'hidden_size']
    input_size: int
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.w_xi = Parameter(Tensor(input_size, hidden_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(Tensor(1, hidden_size))

        # forget gate
        self.w_xf = Parameter(Tensor(input_size, hidden_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(Tensor(1, hidden_size))

        # output gate
        self.w_xo = Parameter(Tensor(input_size, hidden_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(Tensor(1, hidden_size))

        # cell
        self.w_xc = Parameter(Tensor(input_size, hidden_size))
        self.w_hc = Parameter(Tensor(hidden_size, hidden_size))
        self.b_c = Parameter(Tensor(1, hidden_size))

        self.reset_weight()
    
    def reset_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
    
    def check_forward_hidden(self, inputs: Tensor, hx: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor) or not isinstance(hx, Tensor):
            raise RuntimeError(
                "Input has inconsistent type: got {}, expected torch.Tensor".format(
                    type(inputs)))

        if inputs.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden batch size {}".format(
                    inputs.size(0), hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden has inconsistent hidden_size: got {}, expected {}".format(
                    hx.size(1), self.hidden_size))
        
        return hx

    def forward(self, inputs:Tensor, state: Tuple[Tensor, Tensor] = None) \
        -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: [batch, input_size]
            state: ([batch, hidden_size], [batch, hidden_size])
        """
        if state is None:
            h_history = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)
            c_history = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)
        else:
            h_history = self.check_forward_hidden(inputs, state[0]).to(inputs.device)
            c_history = self.check_forward_hidden(inputs, state[1]).to(inputs.device)

        it = torch.sigmoid(inputs@self.w_xi + h_history@self.w_hi + self.b_i)
        ft = torch.sigmoid(inputs@self.w_xf + h_history@self.w_hf + self.b_f)
        ot = torch.sigmoid(inputs@self.w_xo + h_history@self.w_ho + self.b_o)
        cell = torch.tanh(inputs@self.w_xc + h_history@self.w_hc + self.b_c)

        ct = ft * c_history + it * cell
        ht = ot * torch.tanh(ct)

        return (ht, ct)


class LSTM(nn.Module):
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
                    cell = LSTMCell(input_size, hidden_size)
                else:
                    cell = LSTMCell(hidden_size, hidden_size)
                self.cells[i].append(cell)
                setattr(self, 'cell{}_{}'.format(i,j), cell)

    def check_forward_inputs(self, inputs:Tensor) -> None:
        if len(inputs.shape) != 3:
            raise RuntimeError(
                'Input has inconsistent dimension: got {}, expect: 3'.format(
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
            - inputs shape:[seq_len, batch, input_size]
            - h_n shape:[num_layer * num_directions, batch, hidden_size]
            - c_n shape:[num_layer * num_directions, batch, hidden_size]
        
        Outputs: outputs, (h_n, c_n)
            - outputs shape:[seq_len, batch, num_directions * hidden_size]
            - h_n shape:[num_layer * num_directions, batch, hidden_size]
            - c_n shape:[num_layer * num_directions, batch, hidden_size]
        """
        self.check_forward_inputs(inputs)
        if self.batch_first:
            inputs = inputs.permute(1, 0, 2)
            # NOTE inputs: [seq_len, batch, input_size]

        outputs = []
        for direction in range(self.num_directions):
            output = self.each_direction_forward(inputs, direction)       
            outputs.append(output)
        outputs = torch.cat(outputs, dim=2)
        if self.batch_first:
            outputs = outputs.permute(1, 0, 2)
        return outputs, (None,None)


if __name__ == '__main__':
    lstm = LSTM(16, 64, num_layers=2)
    x = torch.randn(32, 8, 16)
    outputs, (hn, cn) = lstm(x)
    print(outputs.shape)
    print(hn.shape)
    print(cn.shape)