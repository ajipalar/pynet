import argparse
from pathlib import Path
import pandas as pd
import sys

"""
Reading and parsing excel files
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="Read all excel files in the directory") 

args = parser.parse_args()



def colnames(p):
    return pd.read_excel(p).columns

def get_xlsx_file_paths(dirpath:Path):
    for f in p.iterdir():
        if f.suffix == '.xlsx':
            yield f
           
def print_all_colnames(path):
   p = Path(args.d)
   for f in p.iterdir():
      if f.suffix == '.xlsx':
          print(colnames(f)) 


def 

if args.d:
   print_all_colnames(args.d)
   
    
