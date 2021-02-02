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
"""Implement transformation for input tensors."""

import torch
import torch.nn as nn

from ._utils import meshgrid, meshgrid_exclude_self

__all__ = ['InputTransformMethod', 'InputTransform']

import enum

__all__ = ['JacEnum']

class JacEnum(enum.Enum):
    """A customized enumeration class, adding helper functions for string-based argument parsing."""

    @classmethod
    def from_string(cls, value):
        value = _canonize_enum_value(value)
        return cls(value)

    @classmethod
    def type_name(cls):
        return cls.__name__

    @classmethod
    def choice_names(cls):
        return list(filter(lambda x: not x.startswith('_'), dir(cls)))

    @classmethod
    def choice_objs(cls):
        return [getattr(cls, name) for name in cls.choice_names()]

    @classmethod
    def choice_values(cls):
        return [getattr(cls, name).value for name in cls.choice_names()]

    @classmethod
    def is_valid(cls, value):
        value = _canonize_enum_value(value)
        return value in cls.choice_values()

    @classmethod
    def assert_valid(cls, value):
        assert cls.is_valid(value), 'Invalid {}: "{}". Supported choices: {}.'.format(
            cls.type_name(), value, ','.join(cls.choice_values()))

    def __jsonify__(self):
        return self.value


def _canonize_enum_value(value):
    if type(value) is str:
        value = value.lower()
    return value

class InputTransformMethod(JacEnum):
  CONCAT = 'concat'
  DIFF = 'diff'
  CMP = 'cmp'

class InputTransform(nn.Module):
  """Transform the unary predicates to binary predicates by operations."""

  def __init__(self, method, exclude_self=True):
    super().__init__()
    self.method = InputTransformMethod.from_string(method)
    self.exclude_self = exclude_self

  def forward(self, inputs):
    assert inputs.dim() == 3

    x, y = meshgrid(inputs, dim=1)

    if self.method is InputTransformMethod.CONCAT:
      combined = torch.cat((x, y), dim=3)
    elif self.method is InputTransformMethod.DIFF:
      combined = x - y
    elif self.method is InputTransformMethod.CMP:
      combined = torch.cat([x < y, x == y, x > y], dim=3)
    else:
      raise ValueError('Unknown input transform method: {}.'.format(
          self.method))

    if self.exclude_self:
      combined = meshgrid_exclude_self(combined, dim=1)
    return combined.float()

  def get_output_dim(self, input_dim):
    if self.method is InputTransformMethod.CONCAT:
      return input_dim * 2
    elif self.method is InputTransformMethod.DIFF:
      return input_dim
    elif self.method is InputTransformMethod.CMP:
      return input_dim * 3
    else:
      raise ValueError('Unknown input transform method: {}.'.format(
          self.method))

  def __repr__(self):
    return '{name}({method}, exclude_self={exclude_self})'.format(
        name=self.__class__.__name__, **self.__dict__)
