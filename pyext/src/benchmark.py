from pathlib import Path
import pyext.src.data_in
from project_paths import benchmarks

def write_example_benchmark():
    """Write an an example benchmark file
    to definen format
    """

    fpath = benchmark / "ppi_benchmark_example.tsv"
    assert fpath.is_file is False, f"{fpath} exists\nAborting"

    with open(fpath, 'w') as f:
        line = "interaction-id\t"
