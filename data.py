import random

import numpy as np
import torch

from utils import *

class RichDataset(torch.utils.data.Dataset):
    def __init__(
        self
        , data
    ):
        super(RichDataset).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    # Гененрирует два случайных семпла, забиваем на индекс
    def __getitem__(self, _):
        idx1 = random.randint(0, self.data.shape[0] - 1)
        idx2 = random.randint(0, self.data.shape[0] - 1)
        idx3 = random.randint(0, self.data.shape[0] - 1)
        return (self.data[idx1], self.data[idx2], self.data[idx3])

# Хотим разбить на куски: dll + вход + веса, и сгенерить noise
# ->: настоящий выход + вход 1, шум 1 + вход 1, шум 2 + вход 2, веса 1, веса 2
class collate_fn_rich:
    def __init__(self, noise_size):
        self.noise_size = noise_size

    # (arr1, arr2)
    def __call__(self, samples):
        batch_size = len(samples)

        full_1 = torch.cat([torch.tensor(t1).unsqueeze(0) for (t1, t2, t3) in samples], dim=0)
        full_2 = torch.cat([torch.tensor(t2).unsqueeze(0) for (t1, t2, t3) in samples], dim=0)
        full_3 = torch.cat([torch.tensor(t3).unsqueeze(0) for (t1, t2, t3) in samples], dim=0)

        input_1 = full_1[:, DLL_DIM:-1]
        input_2 = full_2[:, DLL_DIM:-1]
        input_3 = full_3[:, DLL_DIM:-1]

        w_1 = full_1[:, -1]
        w_2 = full_2[:, -1]
        w_real = full_3[:, -1]

        noise_1 = torch.tensor(np.random.uniform(-1.0, 1.0, size=(batch_size, self.noise_size))).float()
        noise_2 = torch.tensor(np.random.uniform(-1.0, 1.0, size=(batch_size, self.noise_size))).float()

        noised_1 = torch.cat((noise_1, input_1), dim=1)
        noised_2 = torch.cat((noise_2, input_2), dim=1)
        real = full_3[:, :-1]

        return (
            real.to(device)
            , noised_1.to(device)
            , noised_2.to(device)
            , w_real.to(device)
            , w_1.to(device)
            , w_2.to(device)
        )


class SingleDataset(torch.utils.data.Dataset):
    def __init__(
        self
        , data
    ):
        super(SingleDataset).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

class collate_fn_single:
    def __init__(self, noise_size):
        self.noise_size = noise_size

    def __call__(self, samples):
        batch_size = len(samples)

        full = torch.cat([torch.tensor(t).unsqueeze(0) for t in samples], dim=0)

        input_features = full[:, DLL_DIM:-1]

        w = full[:, -1]

        noise = torch.tensor(np.random.normal(size=(batch_size, self.noise_size))).float()

        noised = torch.cat((noise, input_features), dim=1)
        real = full[:, :-1]

        return (
            real.to(device)
            , noised.to(device)
            , w.to(device)
        )
