from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import numpy as np
from typing import List, Union, Tuple
from time import time
import collections
try:
    from itertools import izip_longest
except Exception:
    from itertools import zip_longest as izip_longest

def vector2index_img(vector_img):
    vector_img = np.array(vector_img)
    height, width, nb_channels = vector_img.shape[-3:]
    onehot = np.reshape(vector_img, [height, width, nb_channels])
    index_img = np.zeros([height, width], dtype=np.int32)
    for i in range(nb_channels):
        index_img += 2**i * onehot[:,:,i].astype(np.int32)
    return index_img

def render_index_img(index_img):
    symbols = [" ", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "="]
    symbols = symbols + [str(i) for i in range(64)]
    height, width = index_img.shape
    for y in range(height):
        for x in range(width):
            print(symbols[index_img[y,x]], end="")
        print()

def join_vkb_lists(v1: Tuple[List[np.ndarray]], v2: Tuple[List[np.ndarray]]):
    return [vt1+vt2 for vt1, vt2 in zip(v1, v2)]

def stack_vkb(vkb: Tuple[List[np.ndarray]]):
    return tuple([np.stack(vkb_list, axis=-1) if vkb_list else [] for vkb_list in vkb])

def concat_vkb(v1: Tuple[np.ndarray], v2: Tuple[np.ndarray]):
    # dim=-1 is the nb_predicates dim
    vkb = []
    for vt1, vt2 in zip(v1, v2):
        if isinstance(vt1, np.ndarray) and isinstance(vt2, np.ndarray):
            vkb.append(np.concatenate([vt1, vt2], axis=-1))
        elif isinstance(vt1, np.ndarray):
            vkb.append(vt1)
        elif isinstance(vt2, np.ndarray):
            vkb.append(vt2)
        else:
            vkb.append([])
    return tuple(vkb)

def conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    feature_map_shape = (np.array(input_size, dtype=np.int32)+2*padding-dilation*(kernel_size-1)-1)//stride+1
    return feature_map_shape[0], feature_map_shape[1]

def rotate_vec2d(vec, degrees):
    """
    rotate a vector anti-clockwise
    :param vec:
    :param degrees:
    :return:
    """
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R@vec

def ind_dict2list(dic):
    """
    :param dic: dictionary form object ot index, starting from zero
    :return:
    """
    l = list(range(len(dic)))
    for item, index in dic.items():
        l[index] = item
    return l

def discount(r, discounting):
    discounted_reward = np.zeros_like(r, dtype=np.float32)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * discounting + r[i]
        discounted_reward[i] = G
    return discounted_reward

def normalize(scalars):
    mean, std = np.mean(scalars), np.std(scalars)
    return (scalars - mean)/(std+1e-8)

def unique_list(l):
    if not l:
        return []
    return list(OrderedDict.fromkeys(l))


class TimerCollection():
    def __init__(self):
        self._timers = {}

    def get_timer(self, name):
        if name not in self._timers.keys():
            self._timers[name] = Timer()
        return self._timers[name]

    def reset_all(self):
        for timer in self._timers.values():
            timer.reset()

    def get_statistics(self):
        statistics = {}
        for name, timer in self._timers.items():
            statistics[name] = timer.accumulated_time
        return statistics

    def get_summary_str(self):
        statistics = self.get_statistics()
        if "total" not in statistics:
            statistics["total"] = sum(statistics.values())
        summary_str = ""
        for name, t in statistics.items():
            summary_str += "{} takes {} seconds ({}%);\n".format(name, t, t/statistics["total"])
        return summary_str


class Timer():
    """
    Based on context manager.
    Estimate time spent in the context.
    Query by property of accumulated_time
    """
    def __init__(self):
        self._accumulated_time = 0

    @property
    def accumulated_time(self):
        return self._accumulated_time

    def __enter__(self):
        self._start_time = time()

    def __exit__(self, type, value, traceback):
        current_time = time()
        elapsed = current_time - self._start_time
        self._accumulated_time += elapsed
        return False #re-raise any exceptions

    def reset(self):
        self._accumulated_time = 0


# from https://stackoverflow.com/questions/27890052
def find_shape(seq):
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [find_shape(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in izip_longest(*shapes,
                                                                fillvalue=1))

def fill_array(arr, seq):
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = 0
    else:
        for subarr, subseq in izip_longest(arr, seq, fillvalue=()):
            fill_array(subarr, subseq)


class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

