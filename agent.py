import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Policy(nn.Module):
    def __init__(self, h, w, input_channels, num_output):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_output_size(size, kernel_sizes, strides):
            assert len(kernel_sizes) == len(strides)
            assert kernel_sizes

            output_size = size
            for i in range(len(kernel_sizes)):
                output_size = (output_size - kernel_sizes[i]) // strides[i] + 1

            return output_size

        convw = conv2d_output_size(w, [8, 4, 3], [4, 2, 1])
        convh = conv2d_output_size(h, [8, 4, 3], [4, 2, 1])
        linear_output_size = convh * convw * 64

        self.fc1 = nn.Linear(linear_output_size, 512)

        self.score = nn.Linear(512, num_output)
        self.value_linear = nn.Linear(512, 1)

    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self._conv_forward(x)
        logits = self.score(x)
        value = self.value_linear(x)
        return logits, value.squeeze(1)

    def value(self, x):
        x = self._conv_forward(x)
        value = self.value_linear(x)
        return value.squeeze(1)


def choose_action(action_logits):
    dist = Categorical(logits=action_logits)
    return dist.sample()
