import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

class KnowledgeBaseMLP(nn.Module):
    def __init__(self, obj_n, input_dims
                 , num_actions, use_lstm=False):
        super(KnowledgeBaseMLP, self).__init__()
        input_units = sum([n*obj_n**i for i, n in enumerate(input_dims)])
        self.core = torch.nn.Sequential(nn.Linear(input_units, 256),
                                        torch.nn.ReLU(),
                                        nn.Linear(256, 128))
        self.policy = nn.Linear(128, num_actions)
        self.baseline = nn.Linear(128, 1)
        self.num_actions = num_actions
        self.use_lstm = use_lstm

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, obs, core_state=()):
        T, B, *_ = obs["unary_tensor"].shape
        inputs = [[],
                  torch.flatten(obs["unary_tensor"], 0, 1).float(),
                  torch.flatten(obs["binary_tensor"], 0, 1).float()]
        if "nullary_tensor" in obs and obs["nullary_tensor"] != []:
            inputs[0] = torch.flatten(obs["nullary_tensor"], 0, 1).float()
        else:
            del inputs[0]
        x = torch.cat([torch.flatten(tensor, 1) for tensor in inputs], 1)

        x = self.core(x)
        x = F.relu(x)
        policy_logits = self.policy(x)
        baseline = self.baseline(x)

        #if self.training:
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        #else:
            # Don't sample when testing.
            #action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

