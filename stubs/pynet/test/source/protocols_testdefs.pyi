from functools import partial as partial
from pyext.src.protocols import SupportsLogPDF as SupportsLogPDF, SupportsLogScore as SupportsLogScore, SupportsPDF as SupportsPDF, SupportsRV as SupportsRV, SupportsScore as SupportsScore
from pyext.src.typedefs import Index as Index, PRNGKey as PRNGKey
from typing import Any

IMPpynet_spec: Any
IMPpynet_found: Any

def supports_pdf(x: SupportsPDF) -> None: ...
def supports_log_pdf(x: SupportsLogPDF) -> None: ...
def supports_rv(x: SupportsRV) -> None: ...
def supports_score(x: SupportsScore) -> None: ...
def supports_log_score(x: SupportsLogScore) -> None: ...

class Score:
    @staticmethod
    def score() -> float: ...

class LogScore:
    @staticmethod
    def lscore() -> float: ...

score: Score
lscore: LogScore

class Unsafe:
    @staticmethod
    def unsafe() -> None: ...

class ImplementsUnsafe:
    @staticmethod
    def unsafe() -> None: ...

iu: Any

def unsafe_func(x: Unsafe) -> None: ...
