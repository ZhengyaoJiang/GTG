from environment.box.box_world_env import BoxWorld
import numpy as np

class BoxKBEnv(BoxWorld):
    def __init__(self, n, goal_length, num_distractor, distractor_length, max_steps=10**6,
                 collect_key=True, world=None):
        super(BoxKBEnv, self).__init__(n, goal_length, num_distractor, distractor_length,
                                       max_steps, collect_key, world, False)
        self.nb_entities = 1 + n**2


    def get_zeros_vkb(self, arity):
        return np.zeros([self.nb_entities for _ in range(arity)], dtype=np.float32)

    def get_inventory(self):
        target = self.get_zeros_vkb(1)
        if self.task.target_monster.position:
            target[self.position2index(self.task.target_monster.position)] = 1.0
        return [], [target], []

