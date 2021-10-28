import typing as t
from typing import NewType
from pathlib import Path

DirPath = NewType('DirPath', Path)
FilePath = NewType('FilePath', Path)
AnyPath = NewType('AnyPath', Path)
