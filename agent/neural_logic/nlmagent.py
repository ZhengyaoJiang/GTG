import torch
import torch.nn as nn

from agent.util import conv_output_size
from .layer import LogicMachine
from torch.nn import functional as F
import numpy as np

class NLMAgent(nn.Module):
    def __init__(
            self,
            obj_n,
            depth,
            breadth,
            input_dims,
            action_n,
            output_dims,
            exclude_self=True,
            residual=True,
            io_residual=False,
            recursion=False,
            connections=None,
            observation_shape=None,
            nn_units=0,
            activation=nn.Sigmoid(),
            action_type="propositional"
    ):
        super().__init__()
        self.depth = depth
        self.breadth = breadth
        self.residual = residual
        self.io_residual = io_residual
        self.recursion = recursion
        self.connections = connections
        self.num_actions = action_n
        self.obj_n = obj_n
        input_dims[0] = input_dims[0]+nn_units
        self.breadth = breadth
        output_dims = tuple([output_dims]*3)
        if breadth == 1:
            input_dims = tuple(input_dims[:-1])
            output_dims = tuple(output_dims[:-1])
        if breadth==3:
            input_dims = tuple(input_dims+[0])
            output_dims = tuple(list(output_dims)+[2])
        self.output_dims = output_dims
        if self.residual:
            nlm_input_dims=output_dims
        else:
            nlm_input_dims=input_dims
        self.logic_machine = LogicMachine(depth, breadth, nlm_input_dims, output_dims,
                                          [], exclude_self, residual, io_residual, recursion, connections,
                                          activation)
        if self.residual:
            preprocessing = []
            for i, o in zip(input_dims, output_dims):
                if i == 0:
                    preprocessing.append(None)
                else:
                    preprocessing.append(nn.Linear(i,o))
            self.preprocessing_linears = torch.nn.ModuleList(preprocessing)
        self.baseline_linear = nn.Linear(self.logic_machine.output_dims[0], 1)
        self.action_type = action_type
        if action_type in ["propositional", "move_dir", "move_to", "raw"]:
            self.policy_linear = nn.Linear(self.logic_machine.output_dims[0], action_n)
        elif action_type == "relational":
            self.policy_linear = nn.Linear(self.logic_machine.output_dims[2], 1)
        elif action_type == "move_xy":
            self.policy_linear = nn.Linear(self.logic_machine.output_dims[2], 2)
        self.nn_units = nn_units
        if nn_units>0:
            assert observation_shape
            self.conv = nn.Conv2d(observation_shape[-1], 16, kernel_size=3)
            self.flatten = nn.Flatten()
            unit_n = np.prod(conv_output_size(observation_shape[:-1], 3)) * 16
            self.core = nn.Linear(unit_n, 128)
            self.glob = nn.Linear(128, nn_units)
        #self.set_extreme_parameters()

    @torch.no_grad()
    def set_extreme_parameters(self):
        params = self.parameters()
        for param in params:
            param.data = torch.tensor(np.random.randint(2, size=param.shape)*100.0-50, dtype=torch.float32)


    def forward(self, obs):
        T, B, *_ = obs["unary_tensor"].shape
        inputs = [[],
                  torch.flatten(obs["unary_tensor"], 0, 1).float(),
                  torch.flatten(obs["binary_tensor"], 0, 1).float()]
        if self.breadth == 1:
            inputs = inputs[:-1]
        if self.breadth==3:
            inputs.append([])
        if "nullary_tensor" in obs:
            inputs[0] =  torch.flatten(obs["nullary_tensor"], 0, 1).float()

        device=next(self.parameters()).device

        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(device=device)
            if self.residual:
                if isinstance(self.preprocessing_linears[i], nn.Linear):
                    inputs[i] = self.preprocessing_linears[i](inputs[i])
                else:
                    inputs[i] = torch.zeros([T*B, self.output_dims[i]]).to(device=device)
                inputs[i] = torch.relu(inputs[i])

        evaluations = self.logic_machine.forward(inputs)
        if self.action_type in ["propositional", "move_dir", "move_to", "raw"]:
            policy_logits = self.policy_linear(evaluations[0])
            action_probs = F.softmax(policy_logits, dim=1)
        elif self.action_type == "relational":
            policy_logits = torch.flatten(self.policy_linear(evaluations[2]), start_dim=1)
            action_probs = F.softmax(policy_logits, dim=1)
        elif self.action_type == "move_xy":
            policy_logits = torch.flatten(self.policy_linear(evaluations[2]).permute(0, 3, 1, 2),
                                          start_dim=2)
            marginal_action_probs = F.softmax(policy_logits, dim=2)
            action_probs = torch.einsum("bi,bj->bij", marginal_action_probs[:,0],
                                        marginal_action_probs[:,1])
            action_probs = torch.flatten(action_probs, start_dim=1)
        baseline = self.baseline_linear(evaluations[0])
        if self.training:
            try:
                action = torch.multinomial(action_probs, num_samples=1)
            except Exception as e:
                print(e)
        else:
            # Don't sample when testing.
            action = torch.argmax(action_probs, dim=1)
            #action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)

        policy_logits = torch.log(torch.clamp(action_probs, 1e-9, 1.0)).view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        if self.training:
            return dict(policy_logits=policy_logits, baseline=baseline, action=action)

        else:
            return dict(policy_logits=policy_logits, baseline=baseline, action=action,
                     evaluation=evaluations)
