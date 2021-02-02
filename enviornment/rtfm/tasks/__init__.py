# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .rock_paper_scissors import RockPaperScissors, RockPaperScissorsDev, RockPaperScissorsMed, RockPaperScissorsMedDev, RockPaperScissorsHard, RockPaperScissorsHardDev
from .groups import Groups, GroupsDev, GroupsStationary, GroupsStationaryDev, GroupsSimple, GroupsSimpleDev, GroupsSimpleStationary, GroupsSimpleStationaryDev, GroupsSimpleStationarySingleMonster, GroupsSimpleStationarySingleMonsterDev, GroupsSimpleStationarySingleItem, GroupsSimpleStationarySingleItemDev, GroupsNL, GroupsNLDev, GroupsSimpleNL, GroupsSimpleNLDev, GroupsStationaryNL, GroupsStationaryNLDev, GroupsSimpleStationaryNL
from gym.envs.registration import register as register_env
