"""
Consume search triplets to produce useful search algorithms
"""
import collections
import functools
from functools import partial
import inspect
import itertools
import operator
import os
from typing import Any, Callable, Optional, Sequence, Set, Tuple, TypeVar, List


import jax
import jax.numpy as jnp
import jax.scipy as jsp
import _search_producers as producers
from collections import namedtuple


