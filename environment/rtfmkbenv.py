from environment.rtfm.tasks.groups import Groups, F
from environment.rtfm.dynamics.world_object import Wall
from environment.rtfm.dynamics.monster import HostileMonster, Agent
from environment.rtfm.dynamics.item.base_item import BaseItem
from environment.rtfm.dynamics.element import *
from agent.util import join_vkb_lists, stack_vkb

import gym
import numpy as np


class Spec():
    def __init__(self, id):
        self.id = id


class RTFMEnv(gym.Env):
    def __init__(self, room_size=6):
        self.task = Groups((room_size, room_size), featurizer=F.RelativePosition())
        self.observation_space = gym.spaces.Dict()
        self.observation_space.spaces["image"] = gym.spaces.Box(low=0.0, high=1.0, shape=(room_size, room_size, 4))
        self.action_space = gym.spaces.Discrete(len(self.task.action_space))
        self.image_properties = ["agent", "wall", "monster", "item"]
        # two properties: agent, wall, monster, item
        self.room_size=room_size

        self.world = self.task.world
        self.elements = [Cold.describe(), Fire.describe(), Lightning.describe(), Poison.describe()]
        self.monsters = Groups.monsters
        self.modifiers = Groups.modifiers
        self.groups = Groups.groups
        self.abstract_entities = self.elements+self.monsters+self.modifiers+self.groups
        self.spec = Spec("rtfm")
        self.with_vkb=False


    def get_image(self):
        image = np.zeros([self.room_size, self.room_size, 4], dtype=np.float32)
        for position, entity in self.task.world.map.items():
            x, y = position
            if not entity:
               continue
            entity = list(entity)[0]
            if isinstance(entity, HostileMonster):
                image[y,x,0] = 1.0
            elif isinstance(entity, Agent):
                image[y,x,1] = 1.0
            elif isinstance(entity, Wall):
                image[y,x,2] = 1.0
            elif isinstance(entity, BaseItem):
                image[y,x,3] = 1.0
        return image

    def initilize(self):
        pass

    def get_obs(self):
        obs = dict(image=self.get_image())
        return obs

    def step(self, action):
        _, reward, finish, _ = self.task.step(self.task.action_space[action])
        obs = self.get_obs()
        return obs, reward, finish, ""

    def reset(self):
        _, = self.task.reset()
        self.initilize()
        obs = self.get_obs()
        return obs

class RTFMAbstractEnv(RTFMEnv):
    def __init__(self, room_size=6):
        super(RTFMAbstractEnv, self).__init__(room_size)
        self.nb_physical = self.world.height * self.world.width
        self.nb_entities = self.nb_physical+len(self.abstract_entities)
        self.entity2idx = {entity: self.nb_physical+i for i,entity in enumerate(self.abstract_entities)}
        self.nb_unary = 2+len(self.abstract_entities)
        self.nb_binary = 8
        self.initilize()
        self.with_vkb = True

    def position2index(self, position):
        # position (x, y), where y is inversed
        return position[1]*self.room_size+position[0]

    def get_zeros_vkb(self, arity):
        return np.zeros([self.nb_entities for _ in range(arity)], dtype=np.float32)

    def get_abstract_kb(self):
        # abstract entity class
        abstract_ids = [(entity, self.get_zeros_vkb(1)) for entity in self.abstract_entities]
        for entity, abstract_id in abstract_ids:
            abstract_id[self.entity2idx[entity]] = 1.0

        # beat: modifiers -> elements
        beat = self.get_zeros_vkb(2)
        for element, modifiers in self.task.modifier_assignment:
            for modifier in modifiers:
                beat[self.entity2idx[modifier], self.entity2idx[element.describe()]] = 1.0
        # belong: group -> monsters
        belong = self.get_zeros_vkb(2)
        for group, monsters in self.task.group_assignment:
            for monster in monsters:
                belong[self.entity2idx[group],self.entity2idx[monster]] = 1.0
        # target: group
        target = self.get_zeros_vkb(1)
        target[self.entity2idx[self.task.target_group]] = 1.0
        return [], [target]+[abstract_id for entity, abstract_id in abstract_ids], [beat, belong]

    def get_assignment_kb(self):
        # modifiers -> grid
        # elements -> grid
        # monsters -> grid
        modifiers_assignment = self.get_zeros_vkb(2)
        elements_assignment = self.get_zeros_vkb(2)
        monster_assignment = self.get_zeros_vkb(2)
        has_modifier = self.get_zeros_vkb(2)
        has_element = self.get_zeros_vkb(2)
        has_monster_type = self.get_zeros_vkb(2)

        for position, entity in self.task.world.map.items():
            if not entity:
                continue
            entity = list(entity)[0]
            if isinstance(entity, HostileMonster):
                monster_assignment[self.entity2idx[entity.monster_name], self.position2index(position)] = 1.0
                has_monster_type[self.position2index(position), self.entity2idx[entity.monster_name]] = 1.0
                elements_assignment[self.entity2idx[entity.element.describe()], self.position2index(position)] = 1.0
                has_element[self.position2index(position), self.entity2idx[entity.element.describe()]] = 1.0
            if isinstance(entity, BaseItem):
                modifiers_assignment[self.entity2idx[entity.name.split()[0]], self.position2index(position)] = 1.0
                has_modifier[self.position2index(position), self.entity2idx[entity.name.split()[0]]] = 1.0
        return [], [], [modifiers_assignment, elements_assignment, monster_assignment, has_modifier, has_element, has_monster_type]

    def get_inventory_kb(self):
        inv_modifier = self.get_zeros_vkb(1)
        if self.task.agent.inventory.equipped_items:
            modifier = self.task.agent.inventory.equipped_items[0].name.split()[0]
            inv_modifier[self.entity2idx[modifier]] = 1.0
        return [], [inv_modifier], []

    def get_vkb(self):
        vkb = self.background[:]
        vkb = join_vkb_lists(vkb, self.get_inventory_kb())
        vkb = join_vkb_lists(vkb, self.get_assignment_kb())
        return stack_vkb(vkb)

    def initilize(self):
        kb_abstract = self.get_abstract_kb()
        self.background = kb_abstract

    def get_obs(self):
        return dict(image=self.get_image(), VKB=self.get_vkb())


class RTFMOneHopEnv(RTFMEnv):
    def __init__(self, room_size=6):
        super(RTFMOneHopEnv, self).__init__(room_size)
        self.nb_physical = self.world.height * self.world.width
        self.nb_entities = self.nb_physical
        self.nb_unary = 1
        self.nb_binary = 1
        self.with_vkb = True
        self.adv_map = self.get_adv_map()

    def position2index(self, position):
        # position (x, y), where y is inversed
        return position[1]*self.room_size+position[0]

    def get_zeros_vkb(self, arity):
        return np.zeros([self.nb_entities for _ in range(arity)], dtype=np.float32)

    def get_adv_map(self):
        adv_map = {}
        for element, modifiers in self.task.modifier_assignment:
            for modifier in modifiers:
                adv_map[modifier] = element.describe()
        return adv_map

    def get_beat_rel(self):
        beat_rel = self.get_zeros_vkb(2)
        for monster in self.task.world.monsters:
            if isinstance(monster, HostileMonster):
                for item in self.task.world.items:
                    elemental_adv = list(item.elemental_damage.keys())[0]
                    if monster.element.describe() == elemental_adv.describe():
                        beat_rel[self.position2index(item.position), self.position2index(monster.position)] = 1.0
                if self.task.agent.inventory.equipped_items and self.task.agent.position and monster.position:
                    elemental_adv = list(self.task.agent.elemental_damage.keys())[0]
                    if monster.element.describe() == elemental_adv.describe():
                        beat_rel[self.position2index(self.task.agent.position), self.position2index(monster.position)] = 1.0
        return [], [], [beat_rel]

    def get_target(self):
        target = self.get_zeros_vkb(1)
        if self.task.target_monster.position:
            target[self.position2index(self.task.target_monster.position)] = 1.0
        return [], [target], []

    def get_vkb(self):
        vkb = self.get_beat_rel()
        vkb = join_vkb_lists(vkb, self.get_target())
        return stack_vkb(vkb)

    def get_obs(self):
        obs = dict(image=self.get_image(), VKB=self.get_vkb())
        return obs


