import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class CNNAgent(nn.Module):
    def __init__(self, observation_shape, num_actions, type_code, use_lstm=False, embedding_size=16):
        super(CNNAgent, self).__init__()
        self.type_code=type_code
        nb_conv_layers = int(type_code[0])
        nb_dense_layers = int(type_code[2])
        current_shape = np.array(observation_shape)
        conv_layers = []
        for i in range(nb_conv_layers):
            if type_code[1] == "c":
                conv_layers.append(nn.Conv2d(current_shape[-1], embedding_size, kernel_size=3, padding=1))
            elif type_code[1] == "g":
                conv_layers.append(GraphLikeConv(current_shape[-1], embedding_size, kernel_size=3))
            conv_layers.append(nn.ReLU())
            current_shape[-1] = embedding_size
        self.conv = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()
        unit_n = np.prod(current_shape)

        dense_layers = [nn.Linear(unit_n, 128)]
        for i in range(nb_dense_layers-1):
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Linear(128, 128))
        self.core = nn.Sequential(*dense_layers)
        self.policy = nn.Linear(128, num_actions)
        self.baseline = nn.Linear(128, 1)
        self.num_actions = num_actions
        self.use_lstm = use_lstm

    def forward(self, inputs):
        x = inputs["frame"]*1.0

        T, B, *_ = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        device=next(self.parameters()).device
        x = x.to(device)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.core(x)
        x = F.relu(x)
        policy_logits = self.policy(x)
        baseline = self.baseline(x)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return dict(policy_logits=policy_logits, baseline=baseline, action=action)

class GraphLikeConv(nn.Module):
    def __init__(self, input_feature, output_feature, kernel_size, padding=1):
        super(GraphLikeConv, self).__init__()
        self.positional_lienars = nn.ModuleList([nn.Linear(input_feature, output_feature) for _ in range(kernel_size**2)])
        self.unfold_layer = nn.Unfold(kernel_size=kernel_size, padding=padding)

    def forward(self, inputs):
        N, C, height, width = inputs.shape # N is the batch dimension, C is the channel dimension
        x = self.unfold_layer(inputs).transpose(1, 2) # [N, L, C*kernel], L is the number of blocks
        L = x.shape[1]
        x = self.unfold_layer(inputs).reshape([N, L, C, len(self.positional_lienars)])
        positional_x = []
        for i, linear in enumerate(self.positional_lienars):
            positional_x.append(linear(x[:,:,:,i])) # [N, L, C_out] for each
        x = torch.stack(positional_x).mean(dim=0).transpose(1, 2) # [N, C_out, L]
        return x.reshape([N, -1, height, width])

