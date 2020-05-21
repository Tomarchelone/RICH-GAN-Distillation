import torch
from torch import nn

from utils import *

class StudentGenerator(nn.Module):
    def __init__(
        self
        , noise_size
        , hidden_size
        , num_layers
        , teacher_hidden_size
    ):
        super(StudentGenerator, self).__init__()

        self.dnn = nn.Sequential(*(
            [nn.Linear(noise_size + INPUT_DIM, hidden_size), nn.GELU()]
            + [nn.Linear(hidden_size, hidden_size), nn.GELU()] * (num_layers - 1)
            + [nn.Linear(hidden_size, DLL_DIM)]
        ))

        self.mimic = nn.Linear(hidden_size, teacher_hidden_size)

    def forward(self, noised):
        dll_part = self.dnn(noised)

        return torch.cat((dll_part, noised[:, -INPUT_DIM:]), dim=1)
