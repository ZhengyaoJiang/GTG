from gym import register
from agent import Predicate
from agent.util import ind_dict2list, rotate_vec2d, concat_vkb
from enviornment.box.boxworld_gen import color2index, all_colors
import itertools as itt
from gym_minigrid.minigrid import *

OBJECTS = ind_dict2list(OBJECT_TO_IDX)

class GridObject():
    "object is specified by its location"
    def __init__(self, x, y, object_type=[]):
        self.x = x
        self.y = y
        self.type = object_type
        self.attributes = object_type

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def name(self):
        return str(self.type)+"_"+str(self.x)+str(self.y)


def is_front(obj1, obj2, direction_vec)->bool:
    diff = obj2.pos - obj1.pos
    return diff@direction_vec > 0.1


def is_back(obj1, obj2, direction_vec)->bool:
    diff = obj2.pos - obj1.pos
    return diff@direction_vec < -0.1


def is_left(obj1, obj2, direction_vec)->bool:
    left_vec = rotate_vec2d(direction_vec, -90)
    diff = obj2.pos - obj1.pos
    return diff@left_vec > 0.1


def is_right(obj1, obj2, direction_vec)->bool:
    left_vec = rotate_vec2d(direction_vec, 90)
    diff = obj2.pos - obj1.pos
    return diff@left_vec > 0.1

def is_close(obj1, obj2, direction_vec=None)->bool:
    distance = np.abs(obj1.pos - obj2.pos)
    return np.sum(distance)==1

def is_aligned(obj1, obj2, direction_vec=None)->bool:
    diff = obj2.pos - obj1.pos
    return np.any(diff==0)

def is_top_adj(obj1, obj2, direction_vec=None)->bool:
    return obj1.x==obj2.x and obj1.y==obj2.y+1

def is_left_adj(obj1, obj2, direction_vec=None)->bool:
    return obj1.y==obj2.y and obj1.x==obj2.x-1

def is_top_left_adj(obj1, obj2, direction_vec=None)->bool:
    return obj1.y==obj2.y+1 and obj1.x==obj2.x-1

def is_top_right_adj(obj1, obj2, direction_vec=None)->bool:
    return obj1.y==obj2.y+1 and obj1.x==obj2.x+1

def is_down_adj(obj1, obj2, direction_vec=None)->bool:
    return is_top_adj(obj2, obj1)

def is_right_adj(obj1, obj2, direction_vec=None)->bool:
    return is_left_adj(obj2, obj1)

def is_down_right_adj(obj1, obj2, direction_vec=None)->bool:
    return is_top_left_adj(obj2, obj1)

def is_down_left_adj(obj1, obj2, direction_vec=None)->bool:
    return is_top_right_adj(obj2, obj1)

def top_left(obj1, obj2, direction_vec)->bool:
    return (obj1.x-obj2.x) <= (obj1.y-obj2.y)

def top_right(obj1, obj2, direction_vec)->bool:
    return -(obj1.x-obj2.x) <= (obj1.y-obj2.y)

def down_left(obj1, obj2, direction_vec)->bool:
    return top_right(obj2, obj1, direction_vec)

def down_right(obj1, obj2, direction_vec)->bool:
    return top_left(obj2, obj1, direction_vec)

def fan_top(obj1, obj2, direction_vec)->bool:
    top_left = (obj1.x-obj2.x) <= (obj1.y-obj2.y)
    top_right = -(obj1.x-obj2.x) <= (obj1.y-obj2.y)
    return top_left and top_right

def fan_down(obj1, obj2, direction_vec)->bool:
    return fan_top(obj2, obj1, direction_vec)

def fan_right(obj1, obj2, direction_vec)->bool:
    down_left = (obj1.x-obj2.x) >= (obj1.y-obj2.y)
    top_right = -(obj1.x-obj2.x) <= (obj1.y-obj2.y)
    return down_left and top_right

def fan_left(obj1, obj2, direction_vec)->bool:
    return fan_right(obj2, obj1, direction_vec)


def parse_object(x:int, y:int, feature, type="minigrid")->GridObject:
    """
    :param x:
    :param y:
    :param feature: for MinAtar array of length 3. representing object,color and state
    :return:
    """
    if type == "minigrid":
        obj_type = [OBJECTS[feature[0]]]
        obj = GridObject(x,y, obj_type)
    elif type in ["minatar","rtfm"]:
        if np.all(feature==0.0):
            obj_type = ["empty"]
        else:
            obj_type = []
            for i,f in enumerate(feature):
                if f == 1.0:
                    obj_type.append(f"channel{i}")
        obj = GridObject(x, y, object_type=obj_type)
    elif type == "boxworld":
        obj_type = [str(color2index[tuple(feature)])]
        obj = GridObject(x, y, object_type=obj_type)
    else:
        raise ValueError()
    return obj


class DirectionWrapper(gym.core.ObservationWrapper):
    """
    Add a (agent) direction string for each observation
    """
    def __init__(self, env, type="index"):
        super().__init__(env)
        self.observation_space.spaces["direction"] = spaces.Discrete(4)
        self.type = type

    def observation(self, obs):
        image = obs["image"]
        image[image[:,:,0]==10] = np.array([[10, 0, 0]])
        direction_index = self.env.agent_dir
        if self.type == "onehot":
            dir = np.zeros([4])
            dir[direction_index] = 1.0
        elif self.type == "index":
            dir = direction_index
        else:
            raise ValueError(f"type cannot be {self.type}")
        return {"image":image, "mission":obs["mission"],
                "direction":dir}

def offset2idx_offset(x, y, width):
    return y*width+x


from random import randint
class AbsoluteVKBWrapper(gym.core.ObservationWrapper):
    """
    Add a vkb key-value pair, which represents the state as a vectorised knowledge base.
    Entities are objects in the gird-world, predicates represents the properties of them and relations between them.
    """
    def __init__(self, env, background_id="b0", with_portals=False):
        super().__init__(env)

        background_id = background_id[:2]
        if "MiniGrid" in env.unwrapped.spec.id:
            objs = ["unseen", "empty", "wall", "floor", "ball", "goal", "lava", "agent"]
            self.attributes = objs
            self.nullary_predicates = [Predicate("agentRight", 0), Predicate("agentDown", 0),
                                   Predicate("agentUp", 0), Predicate("agentLeft", 0)]
            self.env_type = "minigrid"
        elif env.unwrapped.spec.id=="minatar":
            self.attributes = env.channels
            self.env_type = "minatar"
            self.nullary_predicates = []
        elif env.unwrapped.spec.id=="rtfm":
            self.attributes = [f"channel{i}" for i in range(len(env.image_properties))]
            self.env_type = "rtfm"
            self.nullary_predicates = []
        elif env.unwrapped.spec.id=="boxworld":
            self.attributes = [str(i) for i in range(len(all_colors))]
            self.env_type = "boxworld"
            self.nullary_predicates = []
        else:
            raise ValueError()
        self.unary_predicates = self.attributes
        self.background_id = background_id
        if background_id in ["b0", "nolocal"]:
            self.rel_deter_func = [is_left, is_right, is_front, is_back, is_aligned, is_close]
        elif background_id == "b1":
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj]
        elif background_id in ["b2", "local"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj]
        elif background_id in ["b3", "all"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, is_aligned, is_close]
        elif background_id == "b4":
            self.rel_deter_func = [is_left, is_right, is_front, is_back]
        elif background_id == "b5":
            self.rel_deter_func = [is_top_adj, is_left_adj, is_down_adj, is_right_adj]
        elif background_id in ["b6", "noremote"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_aligned, is_close]
        elif background_id in ["b7", "noauxi"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back]
        elif background_id in ["b8"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, fan_top, fan_down,
                                   fan_left, fan_right]
        elif background_id in ["b9"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, top_right, top_left,
                                   down_left, down_right]

        self.obj_n = np.prod(env.observation_space["image"].shape[:-1]) #physical eneities
        self.nb_all_entities = self.obj_n
        if self.env_type == "rtfm":
            nb_unary = self.env.nb_unary if self.env.with_vkb else 0
            nb_binary = self.env.nb_binary if self.env.with_vkb else 0
            nb_entities = self.nb_entities if self.env.with_vkb else self.obj_n
            self.obs_shape = [(len(self.nullary_predicates)), (nb_entities, len(self.attributes)+nb_unary),
                              (nb_entities, nb_entities, len(self.rel_deter_func)+nb_binary)]
            self.nb_all_entities = nb_entities
        else:
            self.obs_shape = [(len(self.nullary_predicates)), (self.obj_n, len(self.attributes)),
                           (self.obj_n, self.obj_n, len(self.rel_deter_func))]
        self.spatial_tensors = None
        self.with_portals = with_portals

    def img2vkb(self, img, direction=None):
        img = img.astype(np.int32)
        unary_tensors = [np.zeros(self.obj_n) for _ in range(len(self.unary_predicates))]
        objs = []
        portals = []
        for y, row in enumerate(img):
            for x, pixel in enumerate(row):
                obj = parse_object(x, y, pixel, self.env_type)
                objs.append(obj)
                if self.with_portals and (x, y) in self.portal_pairs:
                    portals.append(obj)

        if self.env_type == "minigrid":
            nullary_tensors = np.zeros(4, dtype=np.float32)
            nullary_tensors[direction] = 1
        else:
            nullary_tensors = []

        for obj_idx, obj in enumerate(objs):
            for p_idx, p in enumerate(self.attributes):
                if p in obj.attributes:
                    unary_tensors[p_idx][obj_idx] = 1.0

        if not self.spatial_tensors:
            self.spatial_tensors = [np.zeros([len(objs), len(objs)]) for _ in range(len(self.rel_deter_func))]
            for obj_idx1, obj1 in enumerate(objs):
                for obj_idx2, obj2 in enumerate(objs):
                    direction_vec = DIR_TO_VEC[3] # Up
                    if portals:
                        if obj2.name == portals[0].name:
                            obj2 = portals[1]
                        elif obj2.name == portals[1].name:
                            obj2 = portals[0]
                    for rel_idx, func in enumerate(self.rel_deter_func):
                        if func(obj1, obj2, direction_vec):
                            self.spatial_tensors[rel_idx][obj_idx1, obj_idx2] = 1.0
        binary_tensors = self.spatial_tensors
        nullary_t, unary_t, binary_t = nullary_tensors, np.stack(unary_tensors, axis=-1), np.stack(binary_tensors, axis=-1)
        if self.env_type == "rtfm" and self.env.with_vkb: # fill in the larger tensor
            nb_unaries = unary_t.shape[-1]
            nb_binaries = binary_t.shape[-1]
            unary = np.zeros([ self.nb_entities, nb_unaries], dtype=np.float32)
            unary[:self.nb_physical,:] = unary_t
            binary = np.zeros([self.nb_entities, self.nb_entities, nb_binaries], dtype=np.float32)
            binary[:self.nb_physical,:self.nb_physical,:] = binary_t
            unary_t = unary
            binary_t = binary
        return nullary_t, unary_t, binary_t

    def observation(self, obs):
        obs = obs.copy()

        if "direction" in obs:
            spatial_VKB = self.img2vkb(obs["image"], obs["direction"])
        else:
            spatial_VKB = self.img2vkb(obs["image"])

        if "VKB" in obs:
            obs["VKB"] = concat_vkb(obs["VKB"], spatial_VKB)
        else:
            obs["VKB"] = spatial_VKB
        return obs


class OneHotFullyObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype='uint8'
        )

    def observation(self, obs):
        img = obs['image']
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype='uint8')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = int(img[i, j, 0])
                color = int(img[i, j, 1])
                state = int(img[i, j, 2])

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {
            'mission': obs['mission'],
            'image': out,
            "direction": obs["direction"]
        }


class PaddingWrapper(gym.core.ObservationWrapper):
    """
    pad the observation into a square field
    """
    def __init__(self, env):
        super().__init__(env)

        self.raw_obs_shape = env.observation_space['image'].shape
        length = max(self.raw_obs_shape[0], self.raw_obs_shape[1])

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(length, length, self.raw_obs_shape[2]),
            dtype='uint8'
        )
        shorter_length = min(self.raw_obs_shape[0], self.raw_obs_shape[1])
        # assume height is shorter
        if shorter_length < length:
            # pad with "unseen"
            self.padding = np.zeros([self.raw_obs_shape[0], length-shorter_length,
                                    self.raw_obs_shape[2]], dtype=np.float32)
        else:
            self.padding = None

    def observation(self, observation):
        if self.padding is not None:
            img = observation["image"]
            img = np.concatenate([self.padding, img], axis=1)
            observation["image"] = img
            return observation
        else:
            return observation


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5
    # Done completing task
    done = 6


class MoveDirActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.field_size = env.observation_space.spaces["image"].shape[:-1]
        self.action_space = gym.spaces.Discrete(8)

    def step(self, action):
        action = int(action)
        if action>3:
            return self.env.step(action-1)
        while True:
            agent_dir = DIR_TO_VEC[self.env.agent_dir]
            if DIR_TO_VEC[action]@agent_dir>0.9:
                return self.env.step(Actions.forward.value)
            elif rotate_vec2d(DIR_TO_VEC[action], 90)@agent_dir>0.9:
                o,r,d,i = self.env.step(Actions.left.value)
            else:
                o,r,d,i = self.env.step(Actions.right.value)
            if d:
                return o, r, d, i


class ProtalWrapper(gym.Wrapper):
    def __init__(self, env, portal_pairs):
        super().__init__(env)
        self.env = env
        if not portal_pairs:
            self.random_portal = True
            self.initialize_portal()
        else:
            self.portal_pairs = portal_pairs


    def step(self, action):
        results = self.env.step(action)
        pos = tuple(self.env.unwrapped.agent_pos)
        if action==2: # forward
            if pos == self.portal_pairs[0]:
                self.unwrapped.agent_pos = self.portal_pairs[1]
            elif pos == self.portal_pairs[1]:
                self.unwrapped.agent_pos = self.portal_pairs[0]
        return results

    def initialize_portal(self):
        self.portal_pairs = ((randint(1, 7), randint(1, 3)),
                             (randint(1, 7), randint(5, 7)))

    def reset(self, **kwargs):
        if self.random_portal:
            self.initialize_portal()
        return super().reset(**kwargs)


class ClosedLavaCrossingEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """
    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        #self.put_obj(Goal(), width - 2, 1)
        v, h = object(), object()

        # Lava rivers or walls specified by direction and position in grid
        #rivers = [(h, j) for j in range(2, width - 2, 2)]
        rivers = [(h, width//2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

register(
    id='MiniGrid-LavaCrossingClosed-v0',
    entry_point='agent.enviornment.minigrid:ClosedLavaCrossingEnv'
)

class MoveToActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.field_size = env.observation_space.spaces["image"].shape[:-1]
        self.action_space = gym.spaces.Discrete(np.prod(self.field_size))

    def action(self, act):
        height, width = self.field_size[0], self.field_size[1]
        target_y = act // width
        target_x = act % width
        agent_x = self.env.agent_pos[0]
        agent_y = self.env.agent_pos[1]
        agent = GridObject(agent_x, agent_y)
        target = GridObject(target_x, target_y)
        direction_vec = DIR_TO_VEC[self.env.agent_dir]
        if is_front(agent, target, direction_vec):
            act = Actions.forward
        elif is_left(agent, target, direction_vec):
            act = Actions.left
        else:
            act = Actions.right
        if not is_aligned(agent, target) or (target_x==agent_x and target_y==agent_y):
            act = Actions.pickup
        return act.value

