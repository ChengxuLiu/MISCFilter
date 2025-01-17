import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch
from torch.autograd import Variable
import torch.nn as nn


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False




def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))


class Module_CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=0.001):
        super(Module_CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon ** 2))


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def moduleNormalize(frame):
    return torch.cat([(frame[:, 0:1, :, :] - 0.4631), (frame[:, 1:2, :, :] - 0.4352), (frame[:, 2:3, :, :] - 0.3990)], 1)


def normalize(x):
    return x * 2.0 - 1.0


def denormalize(x):
    return (x + 1.0) / 2.0


def meshgrid(height, width, grid_min, grid_max):
    x_t = torch.matmul(
        torch.ones(height, 1), torch.linspace(grid_min, grid_max, width).view(1, width))
    y_t = torch.matmul(
        torch.linspace(grid_min, grid_max, height).view(height, 1), torch.ones(1, width))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y