# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import torch
from torch import nn, Tensor

from torch_geometric.data import Data as GeometricData, Batch
from typing import List, Tuple

import logging

logger = logging.getLogger(__name__)


def batch_to_gd(batch: Tensor) -> Tuple[Batch, List[int]]:
    # [B x R x E x E]
    batch_size = batch.shape[0]
    nb_relations = batch.shape[1]
    nb_objects = batch.shape[2]

    assert batch.shape[2] == batch.shape[3]

    x = torch.arange(nb_objects).view(-1, 1)
    i_lst = [x.view(nb_relations, nb_objects, nb_objects) for x in torch.split(batch, 1, dim=0)]

    def to_gd(tensor: Tensor) -> GeometricData:
        nz = torch.nonzero(tensor)
        edge_attr = nz[:, 0]
        # edge_lst = nz[:, 1:].cpu().numpy().tolist()
        # edge_index = torch.LongTensor(list(zip(*edge_lst)))
        edge_index = nz[:, 1:].T
        return GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    batch_data = [to_gd(instance) for instance in i_lst]

    geometric_batch = Batch.from_data_list(batch_data)
    max_node = max(i + 1 for b in batch_data for i in b.x[:, 0].cpu().numpy())
    slices = [max_node for _ in batch_data]
    return geometric_batch, slices
