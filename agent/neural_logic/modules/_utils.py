#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for tensor masking."""

import torch

import torch.autograd as ag

__all__ = ['meshgrid', 'meshgrid_exclude_self', 'exclude_mask', 'mask_value']


import torch
import collections

def concat_shape(*shapes):
  output = []
  for s in shapes:
    if isinstance(s, collections.Sequence):
      output.extend(s)
    else:
      output.append(int(s))
  return tuple(output)

def broadcast(tensor, dim, size):
  if dim < 0:
    dim += tensor.dim()
  assert tensor.size(dim) == 1
  shape = tensor.size()
  return tensor.expand(concat_shape(shape[:dim], size, shape[dim + 1:]))


def meshgrid(input1, input2=None, dim=1):
    """Perform np.meshgrid along given axis. It will generate a new dimension after dim."""
    if input2 is None:
        input2 = input1
    if dim < 0:
        dim += input1.dim()
    n, m = input1.size(dim), input2.size(dim)
    x = broadcast(input1.unsqueeze(dim + 1), dim + 1, m)
    y = broadcast(input2.unsqueeze(dim + 0), dim + 0, n)
    return x, y


def meshgrid_exclude_self(input, dim=1):
    """
    Exclude self from the grid. Specifically, given an array a[i, j] of n * n, it produces
    a new array with size n * (n - 1) where only a[i, j] (i != j) is preserved.
    The operation is performed over dim and dim +1 axes.
    """
    if dim < 0:
        dim += input.dim()

    n = input.size(dim)
    assert n == input.size(dim + 1)

    # exclude self-attention
    rng = torch.arange(0, n, dtype=torch.long, device=input.device)
    rng_n1 = rng.unsqueeze(1).expand((n, n))
    rng_1n = rng.unsqueeze(0).expand((n, n))
    mask_self = (rng_n1 != rng_1n)

    for i in range(dim):
        mask_self.unsqueeze_(0)
    for j in range(input.dim() - dim - 2):
        mask_self.unsqueeze_(-1)
    target_shape = concat_shape(input.size()[:dim], n, n-1, input.size()[dim+2:])

    return input.masked_select(mask_self).view(target_shape)

def exclude_mask(inputs, cnt=2, dim=1):
  """Produce an exclusive mask.

  Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
  a mask with size n * n where only a[i, j] = 1 if and only if (i != j).

  Args:
    inputs: The tensor to be masked.
    cnt: The operation is performed over [dim, dim + cnt) axes.
    dim: The starting dimension for the exclusive mask.

  Returns:
    A mask that make sure the coordinates are mutually exclusive.
  """
  assert cnt > 0
  if dim < 0:
    dim += inputs.dim()
  n = inputs.size(dim)
  for i in range(1, cnt):
    assert n == inputs.size(dim + i)

  rng = torch.arange(0, n, dtype=torch.long, device=inputs.device)
  q = []
  for i in range(cnt):
    p = rng
    for j in range(cnt):
      if i != j:
        p = p.unsqueeze(j)
    p = p.expand((n,) * cnt)
    q.append(p)
  mask = q[0] == q[0]
  # Mutually Exclusive
  for i in range(cnt):
    for j in range(cnt):
      if i != j:
        mask *= q[i] != q[j]
  for i in range(dim):
    mask.unsqueeze_(0)
  for j in range(inputs.dim() - dim - cnt):
    mask.unsqueeze_(-1)

  return mask.expand(inputs.size()).float()


def mask_value(inputs, mask, value):
  assert inputs.size() == mask.size()
  return inputs * mask + value * (1 - mask)
