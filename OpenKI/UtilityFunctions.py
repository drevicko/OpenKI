#   Copyright (c) 2020.  CSIRO Australia.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
from argparse import Namespace
from collections import OrderedDict, Sequence
from copy import deepcopy

import subprocess
from ctypes import Union
from typing import Optional

from OpenKI import logger
from itertools import zip_longest

import torch


def update_args(args, new_args, action_groups, exclude=("Model arguments",), silent=("",), force=(),
                list_arguments=("relation_scorers", "data_variants")):
    """
    Update Namespace args with entries in new_args excluding action groups in 'exclude'.
    Logs updated entries at level INFO and differing entries that aren't updated at level WARNING
    :param args: Namespace to update
    :param new_args: Namespace with new values
    :param action_groups: _action_groups attribute of original argument parser
    :param exclude: tuple of action group names to exclude from the update
    :param silent: do not warn when we can't update these arguments
    :param force: update these values even if new_args has the default value and allows arguments in excluded action
                groups to be updated.
    :param list_arguments: arguments that are lists (new values will be appended)
    :return: Void
    """
    for group in action_groups:
        # if group.title == 'optional arguments':
        #     continue
        for action in group._group_actions:
            dest = action.dest  # name of an attribute
            if dest == "help":
                continue
            if dest not in args:
                logger.warning(f"argument {dest} not found in old args, adding with default value {action.default}")
                setattr(args, dest, action.default)
            new_value = getattr(new_args, dest)
            old_value = getattr(args, dest)
            if new_value == action.default and dest not in force and getattr(args, dest, None) is not None:
                continue  # we can't reset to default values, as we don't know if it's set explicitly
            if old_value != new_value:
                if group.title not in exclude or dest in force:
                    if dest in list_arguments:
                        changed = False
                        for new_scorer in new_value:
                            if new_scorer not in old_value:
                                old_value.append(new_scorer)
                                changed = True
                        if changed:
                            setattr(args, dest, old_value)
                            logger.info(f"Appending {group.title} argument {dest} to {old_value} with {new_value}")
                    else:
                        setattr(args, dest, new_value)
                        logger.info(f"updating  {group.title} argument {dest} from {old_value} to {new_value}")
                elif dest not in silent:
                    logger.warning(f"can't update {group.title} argument {dest} from {old_value} to {new_value}! "
                                   f"It's built into the model!")


def grouper(iterable, n, fill_value=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


def refuse_cuda(self, is_cuda=True):
    """
    Dummy function for monkey patching cuda() on an nn.Parameter (eg: an embedding), forcing it to not respond to
    calls to `cuda()`. You still need to suitably process input tensors such that the module receives cpu inputs
    and process it's outputs such that subsequent processing receives gpu.
    To monkeypatch a parameter `self.weight`, use `self.weight.cuda = refuse_cuda.__get__(self.weight)`.
    A `forward()` method like the following may also be appropriate:
        def forward_cpu(self, input: torch.Tensor) -> torch.Tensor:
            output = super().forward(input.cpu())
            if self.is_cuda_:
                output = output.cuda()
            return output
    :param self:
    :param is_cuda: The value cuda() is supposed do be set to (eg: set cuda() on model outputs to this)
    :return: self
    """
    self.is_cuda_ = is_cuda
    return self


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def list_param_devices(mod: torch.nn.Module):
    module_list = [str(mod)]
    for submodule in mod.modules():
        module_list.append(f"{type(submodule)} with params on {set(p.device for p in submodule.parameters())}")
    return '\n'.join(module_list)


def diff_args(old_args, new_args):
    def finish_up(an_iter, other_v, msg):
        for k, v in an_iter:
            if msg == "added":
                yield k, other_v, v, msg
            else:
                yield k, v, other_v, msg
    new_arg_iter, old_arg_iter = tuple(iter(sorted(vars(args).items())) for args in (new_args, old_args))
    for (k_new, v_new), (k_old, v_old) in zip_longest(new_arg_iter, old_arg_iter, fillvalue=(None, None)):
        if k_new is None:
            if k_old is not None:
                yield k_old, v_old, None, "removed"
            yield from finish_up(old_arg_iter, None, "removed")
            return
        if k_old is None:
            if k_new is not None:
                yield k_new, None, v_new, "added"
            yield from finish_up(new_arg_iter, None, "added")
            return
        while k_new != k_old:
            while k_new < k_old:
                yield k_new, None, v_new, "added"
                try:
                    k_new, v_new = next(new_arg_iter)
                except StopIteration:
                    yield k_old, v_old, None, "removed"
                    yield from finish_up(old_arg_iter, None, "removed")
                    return
            while k_old < k_new:
                yield k_old, v_old, None, "removed"
                try:
                    k_old, v_old = next(old_arg_iter)
                except StopIteration:
                    yield k_new, None, v_new, "added"
                    yield from finish_up(new_arg_iter, None, "added")
                    return
        # k_new == k_old
        if v_new != v_old:
            yield k_new, v_old, v_new, "changed"


class OrderedDefaultDict(OrderedDict, Sequence):
    def __init__(self, default_factory=None, index_key=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory
        self.index_key = index_key
        if index_key is None:
            self.indices = {}
        elif default_factory is None:
            self.indices = None
        elif type(self.default_factory()) is dict:
            self.indices = None
        else:
            self.indices = {}

    def __getitem__(self, key):
        item = self.get(key)
        if item is None:
            val = self.default_factory()
            if self.indices is None:
                val[self.index_key] = len(self)
            else:
                self.indices[key] = len(self)
            self[key] = val
            return val
        else:
            return item

    def index(self, key):
        if self.indices is None:
            return self[key][self.index_key]
        else:
            return self.indices[key]

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               super().__repr__())


def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]
    return int(out.partition(b' ')[0])


if __name__ == "__main__":
    def compare_args(new, old):
        print(f"old {old}")
        print(f"new {new}")
        for dif in diff_args(old, new):
            print(dif)
        print()


    args_new = Namespace(**{"a": 1, "b": 2})
    args_old = deepcopy(args_new)
    compare_args(args_new, args_old)

    args_new.a = 2
    args_new.c = 3
    args_old.d = 4
    compare_args(args_new, args_old)

    args_new.a = 1
    args_new.g = 5
    args_new.h = 6
    compare_args(args_new, args_old)

    args_new.d = 4
    compare_args(args_new, args_old)

    args_old.c = 3
    args_old.e = 9
    args_old.g = 9
    args_old.h = 6
    args_old.j = 9
    args_old.k = 9
    compare_args(args_new, args_old)
