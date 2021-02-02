from agent.geometric.util import batch_to_gd
from torch_geometric.nn import RGCNConv
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from typing import List

def parse_code(net_code: str):
    """
    :param net_code: format <a>g[m]<b>f
    """
    assert net_code[1]=="g"
    assert net_code[-1]=="f"
    nb_gnn_layers = int(net_code[0])
    nb_dense_layers = int(net_code[-2])
    is_max = True if net_code[2] == "m" else False
    return nb_gnn_layers, nb_dense_layers, is_max

class GNNAgent(nn.Module):
    def __init__(self, obj_n: int, action_n: int, input_dims:List[int], type: str, embedding_size=16, net_code="2g0f", mp_rounds=1):
        super().__init__()
        nb_edge_types = input_dims[2]
        nb_layers, nb_dense_layers, self.max_reduce = parse_code(net_code)
        self.embedding_linear = nn.Linear(input_dims[1], embedding_size)
        gnn_layers = []
        for i in range(nb_layers):
            gnn_layers.append(RGCNConv(embedding_size, embedding_size, nb_edge_types))
        self.gnn_layers = nn.ModuleList(gnn_layers)
        dense_layers = []
        for i in range(nb_dense_layers):
            if i == 0:
                if self.max_reduce:
                    dense_layers.append(nn.Linear(embedding_size, 128))
                else:
                    dense_layers.append(nn.Linear(embedding_size*obj_n, 128))
            else:
                dense_layers.append(nn.Linear(128, 128))
            dense_layers.append(nn.ReLU())
        self.dense = nn.Sequential(*dense_layers)
        self.num_actions = action_n
        if nb_dense_layers == 0:
            self.policy_linear = nn.Linear(embedding_size, self.num_actions)
            self.baseline_linear = nn.Linear(embedding_size, 1)
        else:
            self.policy_linear = nn.Linear(128, self.num_actions)
            self.baseline_linear = nn.Linear(128, 1)
        self.mp_rounds = mp_rounds
        self.nb_dense_layers = nb_dense_layers

    def forward(self, obs, core_state=()):
        T, B, *_ = obs["unary_tensor"].shape
        device=next(self.parameters()).device
        inputs = [[],
                  torch.flatten(obs["unary_tensor"], 0, 1).float(),
                  torch.flatten(obs["binary_tensor"], 0, 1).permute(0,3,1,2).float()]
        if "nullary_tensor" in obs:
            inputs[0] =  torch.flatten(obs["nullary_tensor"], 0, 1).float()
        for i in [1,2]:
            inputs[i] = inputs[i].to(device=device)
        adj_matrices = inputs[2]
        gd, slices = batch_to_gd(adj_matrices)
        embedds = torch.flatten(inputs[1], 0, 1)
        embedds = self.embedding_linear(embedds)
        for layer in self.gnn_layers:
            for _ in range(self.mp_rounds):
                embedds = layer.forward(embedds, gd.edge_index, gd.edge_attr)
                embedds = torch.relu(embedds)
        chunks = torch.split(embedds, slices, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        x = torch.cat(chunks, dim=0)
        if self.max_reduce:
            x, _ = torch.max(x, dim=1)
        else:
            x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.dense(x)
        policy_logits = self.policy_linear(x)
        baseline = self.baseline_linear(x)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        if self.training:
            return dict(policy_logits=policy_logits, baseline=baseline, action=action)
        else:
            return dict(policy_logits=policy_logits, baseline=baseline, action=action)

