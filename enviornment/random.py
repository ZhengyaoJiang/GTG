import gym
import numpy as np
from gym import spaces

class Spec():
    def __init__(self, id):
        self.id = id

class RandomEnv(gym.Env):
    def __init__(self):
        super(RandomEnv, self).__init__()
        self.action_space = spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict()
        self.observation_space.spaces["image"] = gym.spaces.Box(low=0.0, high=1.0, shape=(10,10,10))
        self.channels = [f"channel{i}" for i in range(10)]
        self.spec = Spec("random")
        self._step = 0

    def _next_observation(self):
        return np.random.randint(2, size=(10, 10, 10)).astype(np.float32)

    def step(self, action):
        # Execute one time step within the environment
        self._step += 1
        done = True if self._step == 10 else False
        return {"image":self._next_observation()}, 0, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self._step = 0
        return {"image":self._next_observation()}
