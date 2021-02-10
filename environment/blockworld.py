from agent import Atom, Clause, Predicate
import numpy as np
import copy
from random import choice, random, Random
from typing import Union,List,Dict
from collections import namedtuple
import gym


class Spec():
    def __init__(self, id):
        self.id = id

class BlockEnv(gym.Env):
    def __init__(self, name, nb_blocks=4, variation=None, rand_env=False, state_block_spec=True):
        super().__init__()
        if rand_env:
            initial_state = generate_initial(self.seed(), nb_blocks)
            if name == "block-on":
                self.env = On(block_n=nb_blocks, initial_state=initial_state, state_block_spec=state_block_spec)
            elif name == "block-stack":
                self.env = Stack(block_n=nb_blocks, initial_state=initial_state, state_block_spec=state_block_spec)
            elif name == "block-unstack":
                self.env = Unstack(block_n=nb_blocks, initial_state=initial_state, state_block_spec=state_block_spec)
        else:
            if name == "block-on":
                self.env = On(block_n=nb_blocks, state_block_spec=state_block_spec)
            elif name == "block-stack":
                self.env = Stack(block_n=nb_blocks, state_block_spec=state_block_spec)
            elif name == "block-unstack":
                self.env = Unstack(block_n=nb_blocks, state_block_spec=state_block_spec)
        if isinstance(variation, str):
            self.env = self.env.vary(variation)
        elif isinstance(variation, int) and variation>0:
            self.env = self.env.vary(self.env.all_variations[variation-1])
        self.observation_space = gym.spaces.Dict()
        self.observation_space.spaces["image"] = gym.spaces.Box(low=0.0, high=1.0,
                                                                shape=self.env.statetensor.shape)
        self.action_space = gym.spaces.Discrete(self.env.action_n)
        self.spec = Spec(name)
        self.channels = [f"channel{i}" for i in range(self.env.statetensor.shape[-1])]
        self.nb_blocks = nb_blocks
        self.rand_env = rand_env

    def reset(self):
        initial_state=generate_initial(self.seed(), nb_blocks=self.nb_blocks) if self.rand_env else None
        self.env.reset(initial_state)
        return {"image": self.env.statetensor.astype(np.float32)}

    def render(self, mode='human'):
        raise NotImplementedError()

    def step(self, action):
        r, done = self.env.next_step(self.env.all_actions[action])
        return {"image": self.env.statetensor.astype(np.float32)}, r, done, ""

class BlockActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.block_env = env.env
        self.action_space = gym.spaces.Discrete(len(self.block_env.all_blocks) ** 2)

    def action(self, act):
        return act+1

class GridActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.block_env = env.env
        self.field_size = env.observation_space.spaces["image"].shape[:-1]
        self.action_space = gym.spaces.Discrete(np.prod(self.field_size)**2)

    def get_obj_idx(self, position_index):
        height, width = self.field_size[0], self.field_size[1]
        y = position_index // height
        x = position_index % height
        onehot = self.block_env.state2tensor(self.block_env.state, block_spec=True)[y, x]
        np.argmax(onehot)
        return self.block_env.onehot2index(onehot)

    def action(self, act):
        height, width = self.field_size[0], self.field_size[1]
        nb_grids = height * width
        from_index = act // nb_grids
        from_obj = self.get_obj_idx(from_index)
        to_index = act % nb_grids
        to_obj = self.get_obj_idx(to_index)
        if not from_obj or not to_obj:
            return 0 #No action
        else:
            return from_obj*len(self.block_env.all_blocks)+to_obj

class SymbolicEnvironment(object):
    def __init__(self, background, initial_state, actions):
        '''
        :param language_frame
        :param background: list of atoms, the background knowledge
        '''
        self.background = background
        self._state = copy.deepcopy(initial_state)
        self.initial_state = copy.deepcopy(initial_state)
        self.actions = actions
        self.acc_reward = 0
        self.step = 0

    def reset(self, initial_state):
        self.acc_reward = 0
        self.step = 0
        if not initial_state:
            self._state = copy.deepcopy(self.initial_state)
        else:
            self._state = initial_state


ON = Predicate("on", 2)
TOP = Predicate("top", 1)
MOVE = Predicate("move", 2)
INI_STATE = [["a", "b", "c", "d"]]
INI_STATE2 = [["a"], ["b"], ["c"], ["d"]]
#INI_STATE2 = [["a"], ["b"]]
FLOOR = Predicate("floor", 1)
BLOCK = Predicate("block", 1)
CLEAR = Predicate("clear", 1)

def generate_initial(seed, nb_blocks):
    blocks = list(string.ascii_lowercase)[:nb_blocks]
    rand = Random(seed)
    state = [[] for _ in range(nb_blocks)]
    rand.shuffle(blocks)
    for block in blocks:
        state[rand.randrange(nb_blocks)].append(block)
    return state

import string
class BlockWorld(SymbolicEnvironment):
    """
    state is represented as a list of lists
    """
    def __init__(self, initial_state=INI_STATE, additional_predicates=(), background=(), block_n=4,
                 all_block=False, neural_Pa=False, state_encoder="relative", state_block_spec=False):
        actions = [MOVE]
        self.max_step = 50
        self._block_encoding = {"floor":0, "a":1, "b": 2, "c":3, "d":4, "e": 5, "f":6, "g":7}
        self._state_shape = [block_n+1, block_n+1, 2]
        self._all_blocks = ["floor"]+list(string.ascii_lowercase)[:block_n]
        self.neural_Pa = neural_Pa
        if neural_Pa:
            actions = []
        #self.language = LanguageFrame(actions, extensional=[ON,FLOOR, TOP]+list(additional_predicates),
        #                          constants=self._all_blocks)
        self._additional_predicates = additional_predicates
        background = list(background)
        background.append(Atom(FLOOR, ["floor"]))
        #background.extend([Atom(BLOCK, [b]) for b in list(string.ascii_lowercase)[:block_n]])
        super(BlockWorld, self).__init__(background, initial_state, actions)
        self._block_n = block_n
        self.block_spec = state_block_spec

    @property
    def all_blocks(self):
        return self._all_blocks

    def state2embeddings(self, state):
        if isinstance(state, tuple):
            state = self.state2tensor(state)
        return self._state_encoder.state2embeddings(state)

    def onehot2index(self, onehot):
        if np.all(onehot == 0):
            return None
        else:
            return np.argmax(onehot)

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def state_dim(self):
        return np.prod(self._state_shape)

    def clean_empty_stack(self):
        self._state = [stack for stack in self._state if stack]

    @property
    def all_actions(self):
        return [None]+[Atom(MOVE, [a, b]) for a in self._all_blocks for b in self._all_blocks]

    @property
    def state(self):
        return tuple([tuple(stack) for stack in self._state])

    @property
    def statetensor(self):
        return self.state2tensor(self.state)

    def next_step(self, action):
        """
        :param action: action is a ground atom
        :return:
        """

        self.step+=1
        reward, finished = self.get_reward()
        self.acc_reward += reward

        self.clean_empty_stack()
        if action == None:
            return reward, finished
        block1, block2 = action.terms
        if finished and reward<1:
            self._state = [[]]
            return reward, finished
        for stack1 in self._state:
            if stack1[-1] == block1:
                for stack2 in self._state:
                    if stack2[-1] == block2:
                        del stack1[-1]
                        stack2.append(block1)
                        return reward, finished
        if block2 == "floor":
            for stack1 in self._state:
                if stack1[-1] == block1 and len(stack1)>1:
                    del stack1[-1]
                    self._state.append([block1])
                    return reward, finished
        return reward, finished

    @property
    def action_n(self):
        return len(self.all_actions)

    def state2tensor(self, state):
        block_spec = self.block_spec
        if block_spec:
            width = self._state_shape[0]
            matrix = np.zeros([width for _ in range(3)], dtype=np.float32)
        else:
            matrix = np.zeros(self._state_shape, dtype=np.float32)
        for x, stack in enumerate(state):
            for y, block in enumerate(stack):
                if block_spec:
                    matrix[y+1][x][self._block_encoding[block]] = 1.0
                else:
                    matrix[y+1][x][1] = 1.0
        matrix[0, :, 0] = 1.0
        return matrix

    def state2vector(self, state):
        return self.state2tensor(state).flatten()

    def state2atoms(self, state):
        atoms = set()
        for stack in state:
            if len(stack)>0:
                atoms.add(Atom(ON, [stack[0], "floor"]))
                atoms.add(Atom(TOP, [stack[-1]]))
            for i in range(len(stack)-1):
                atoms.add(Atom(ON, [stack[i+1], stack[i]]))
        return atoms

    def get_reward(self):
        pass

class BlockVKBWarpper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.blocks = env.unwrapped.env.all_blocks
        self.obj_n = len(self.blocks)
        self.obs_shape = [(0), (self.obj_n, 2),
                          (self.obj_n, self.obj_n, 1)]

    def img2vkb(self, img):
        is_floor = np.zeros([self.obj_n])
        is_floor[0] = 1.0
        on = np.zeros([self.obj_n, self.obj_n])
        top = np.zeros([self.obj_n])
        for y, row in enumerate(img):
            for x, feature in enumerate(row):
                if np.any(feature > 0.0) and y > 0:
                    block_id = np.argmax(feature)
                    if y == 1:
                        on[block_id, 0] = 1.0
                    else:
                        below_block_id = np.argmax(img[y-1,x])
                        on[block_id, below_block_id] = 1.0
                    if y == self.obj_n-1:
                        top[block_id] = 1.0
                    elif np.all(img[y+1,x] == 0.0): #no blocks above
                        top[block_id] = 1.0
                    block_id += 1
        return [], np.stack([is_floor, top], axis=-1), np.stack([on], axis=-1)


    def observation(self, obs):
        obs = obs.copy()
        obs["VKB"] = self.img2vkb(obs["image"])
        return obs

class Unstack(BlockWorld):
    all_variations = ("swap top 2","2 columns", "5 blocks",
                      "6 blocks", "7 blocks")

    nn_variations = ("swap top 2", "2 columns")
    def get_reward(self):
        if self.step >= self.max_step:
            return -0.0, True
        for stack in self._state:
            if len(stack) > 1:
                return -0.02, False
        return 1.0, True

    def vary(self, type, all_block=False):
        block_n = self._block_n
        if type=="swap top 2":
            initial_state=[["a", "b", "d", "c"]]
        elif type=="2 columns":
            initial_state=[["b", "a"], ["c", "d"]]
        elif type=="5 blocks":
            initial_state=[["a", "b", "c", "d", "e"]]
            block_n = 5
        elif type=="6 blocks":
            initial_state=[["a", "b", "c", "d", "e", "f"]]
            block_n = 6
        elif type=="7 blocks":
            initial_state=[["a", "b", "c", "d", "e", "f", "g"]]
            block_n = 7
        else:
            raise ValueError
        return Unstack(initial_state, self._additional_predicates, self.background, block_n, all_block=all_block,
                       neural_Pa=self.neural_Pa)


class Stack(BlockWorld):

    all_variations = ("swap right 2","2 columns", "5 blocks",
                      "6 blocks", "7 blocks")
    nn_variations = ("swap right 2", "2 columns")

    def __init__(self, initial_state=INI_STATE2, block_n=4, all_block=False, neural_Pa=False,
                 state_encoder="relative", state_block_spec=False):
        super(Stack, self).__init__(initial_state, block_n=block_n, all_block=all_block, neural_Pa=neural_Pa,
                                    state_encoder=state_encoder, state_block_spec=state_block_spec)

    def get_reward(self):
        if self.step >= self.max_step:
            return -0.0, True
        for stack in self._state:
            if len(stack) == self._block_n:
                return 1.0, True
        return -0.02, False

    def vary(self, type, all_block=False):
        block_n = self._block_n
        if type=="swap right 2":
            initial_state=[["a"], ["b"], ["d"], ["c"]]
        elif type=="2 columns":
            initial_state=[["b", "a"], ["c", "d"]]
        elif type=="5 blocks":
            initial_state=[["a"], ["b"], ["c"], ["d"], ["e"]]
            block_n = 5
        elif type=="6 blocks":
            initial_state=[["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]
            block_n = 6
        elif type=="7 blocks":
            initial_state=[["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"]]
            block_n = 7
        else:
            raise ValueError
        return Stack(initial_state, block_n, all_block=all_block, neural_Pa=self.neural_Pa)


GOAL_ON = Predicate("goal_on", 2)
class On(BlockWorld):
    all_variations = ("swap top 2","swap middle 2", "5 blocks",
                      "6 blocks", "7 blocks")
    nn_variations = ("swap top 2", "swap middle 2")
    def __init__(self, initial_state=INI_STATE, goal_state=Atom(GOAL_ON, ["a", "b"]), block_n=4,
                 all_block=False, neural_Pa=False, state_encoder="relative", state_block_spec=False):
        super(On, self).__init__(initial_state, additional_predicates=[GOAL_ON],
                                 background=[goal_state], block_n=block_n, all_block=all_block,
                                 neural_Pa=neural_Pa, state_encoder=state_encoder, state_block_spec=state_block_spec)
        self.goal_state = goal_state

    def get_reward(self):
        if self.step >= self.max_step:
            return 0.0, True
        if Atom(ON, self.goal_state.terms) in self.state2atoms(self._state):
            return 1.0, True
        return -0.02, False

    def vary(self, type, all_block=False):
        block_n = self._block_n
        if type=="swap top 2":
            initial_state=[["a", "b", "d", "c"]]
        elif type=="swap middle 2":
            initial_state=[["a", "c", "b", "d"]]
        elif type=="5 blocks":
            initial_state=[["a", "b", "c", "d", "e"]]
            block_n = 5
        elif type=="6 blocks":
            initial_state=[["a", "b", "c", "d", "e", "f"]]
            block_n = 6
        elif type=="7 blocks":
            initial_state=[["a", "b", "c", "d", "e", "f", "g"]]
            block_n = 7
        else:
            raise ValueError
        return On(initial_state, block_n=block_n, all_block=all_block, neural_Pa=self.neural_Pa)
