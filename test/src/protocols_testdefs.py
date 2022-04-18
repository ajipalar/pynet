from __future__ import print_function
import IMP.test
import IMP.algebra
import importlib.util

IMPpynet_spec = importlib.util.find_spec("IMP.pynet")
IMPpynet_found = IMPpynet_spec is not None

try:
    from IMP.pynet.protocols import (
        SupportsPDF,
        SupportsLogPDF,
        SupportsRV,
        SupportsScore,
        SupportsLogScore,
    )
    import IMP.pynet.distributions as dist
    from IMP.pynet.typedefs import Index, PRNGKey
except ModuleNotFoundError:
    from pyext.src.protocols import (
        SupportsPDF,
        SupportsLogPDF,
        SupportsRV,
        SupportsScore,
        SupportsLogScore,
    )
    import pyext.src.distributions as dist
    from pyext.src.typedefs import Index, PRNGKey

import io
import jax
import jax.numpy as jnp
import math
import numpy as np
from functools import partial
from typing import Union, Any, Callable, Protocol

"""
Testing of protocols is done by mypy and not at runtime
"""


def supports_pdf(x: SupportsPDF) -> None:
    ...


def supports_log_pdf(x: SupportsLogPDF) -> None:
    ...


def supports_rv(x: SupportsRV) -> None:
    ...


def supports_score(x: SupportsScore) -> None:
    ...


def supports_log_score(x: SupportsLogScore) -> None:
    ...


class Score:
    @staticmethod
    def score() -> float:
        ...


class LogScore:
    @staticmethod
    def lscore() -> float:
        ...


supports_pdf(dist.norm)
supports_log_pdf(dist.norm)

score: Score = Score()
lscore: LogScore = LogScore()

supports_score(score)
supports_log_score(lscore)

# MYPY FAIL

supports_pdf(score)
supports_log_pdf(score)
supports_score(dist.norm)
supports_log_score(dist.norm)


class Unsafe(Protocol):
    @staticmethod
    def unsafe() -> None:
        ...


class ImplementsUnsafe:
    @staticmethod
    def unsafe() -> None:
        ...


iu = ImplementsUnsafe()


def unsafe_func(x: Unsafe) -> None:
    ...


unsafe_func(iu)
unsafe_func(score)
