from environment.minatar import Environment
from .minigrid import *
import gym
import numpy as np

class Spec():
    def __init__(self, id):
        self.id = id

class MinAtarEnv(gym.Env):
    def __init__(self, name, sticky_action_prob):
        super().__init__()
        self.maximum_steps = 5000
        self.env = Environment(name, sticky_action_prob)
        self.observation_space = gym.spaces.Dict()
        self.observation_space.spaces["image"] = gym.spaces.Box(low=0.0, high=1.0, shape=self.env.state_shape())
        self.action_space = gym.spaces.Discrete(self.env.num_actions())
        self.spec = Spec("minatar")
        self.channels = [f"channel{i}" for i in range(len(self.env.env.channels))]
        self.steps = 0

    @property
    def agent_position(self):
        """
        :return: row_n, column_b
        """

    def reset(self):
        self.env.reset()
        self.steps = 0
        return {"image": self.env.state().astype(np.float32)}

    def render(self, mode='human'):
        self.env.display_state()

    def step(self, action):
        r, done = self.env.act(action)
        self.steps += 1
        if self.steps >= self.maximum_steps:
            done = True
        return {"image": self.env.state().astype(np.float32)}, r, done, ""
