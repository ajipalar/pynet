import argparse
from pathlib import Path
import pandas as pd
import sys

"""
Reading and parsing excel files
pf pure function. Function has no side effects
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="Read all excel files in the directory") 

args = parser.parse_args()



def colnames(p):
    return pd.read_excel(p).columns

def gen_xlsx_fpaths(p):
    """
    :param p pathlib path
    returns a generator
    """
    for f in p.iterdir():
        if f.suffix == '.xlsx':
            yield f

def gen_sheets(fpath):
    """
    :param fpath file path
    returns a generator of sheetnames
    """
    ef = pd.ExcelFile(fpath)
    for sheet in ef.sheet_names:
        yield sheet

def tree_excel(p):
    for xlsx in get_xlsx_file_paths(p):
        print(f'{xlsx}')
        for sheet in get_sheets(xlsx):
            print(f'\t{sheet}')

def condf(f, condition, *args):
    """
    :f funciton
    :condition 
    :*args args for f
    Executes f if condition
    """
    return f(*args) if condition else None
 

def treef(p, f=lambda pathy : None, *args): 
    """
    Recursivley applies a function to files in a directory like linux tree
    """
    for pathy in p.iterdir():
        if pathy.is_file():
            f(*args)
        if pathy.is_dir():
            treef(pathy, f, *args)




def print_all_colnames(path):
   p = Path(args.d)
   for f in p.iterdir():
      if f.suffix == '.xlsx':
          print(colnames(f)) 

if args.d:
   print_all_colnames(args.d)
   
    
