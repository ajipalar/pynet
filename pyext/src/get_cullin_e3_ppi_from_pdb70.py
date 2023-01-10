"""
Run this script from the project home directory as
python -m pyext.src.data.get_cullin_e3_from_pdb70
"""
import click
import numpy
import pandas
import subprocess
from enum import Enum
from pathlib import Path

def main():
    #0 Do some checking
    step0()

    #1 Get the list of cullin UniProt Accession ID's or GIDs

    #2 Get a representative sequence for each of the ids including viral sequences

    #3 Query pdb70 for PDB files with a sequence within threshold, writing these files to an interm directory

    #4 Count the pairwise interactions among the benchmark set, writing these data to an interim csv  

class PyPath(Enum):
    """
    Global immutable path variables
    """

    home = Path(".")
    pyext = home / "pyext"
    src = pyext / "src"
    data = home / "data"
    raw = data / "raw"
    interim = data / "interim"
    cullin_e3_ligase = raw / "cullin_e3_ligase"


def step0():
    for p in [member.value for member in PyPath]:
        assert p.is_dir()

def s1():
    """
    Get the list of cullin UniProt Accession ID's or GIDs
    """
    ...

def s2():
    """
    Get a representative sequence for each of the ids including viral sequences
    """
    ...

def s3():
    """
    Query pdb70 for PDB files with a sequence within threshold, writing these to an interim directory
    """
    ...

def s4():
    """
    Count the pairwise interactions among the benchmark set, writing these data to an interm csv
    """



if __name__ == "__main__":
    main()
