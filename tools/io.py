import argparse
import json
from pathlib import Path
import pandas as pd
import sys
import typing as t
from typing import Dict, Iterable, Iterator, List, Tuple

#Custom modules
import tools.predicates as pred
from net.typedefs import AnyPath, FilePath, DirPath


"""
Reading and parsing excel files
pf pure function. Function has no side effects
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="Read all excel files in the directory") 

args = parser.parse_args()

### Filters paths using predicates ###
def pathsT(paths: Iterator[FilePath], T: str) -> Iterator[FilePath]:
    """Generic for filtering paths functions
    T: file, xlsx, excel, json, txt
    """
    predf = eval(f'pred.is{T}') 
    return filter(predf, paths)

def fpaths(paths: Iterator[AnyPath]) -> Iterator[FilePath]:
    return filter(pred.isfile, paths)

def xlsxpaths(paths: Iterator[FilePath]) -> Iterator[FilePath]:
    return filter(pred.isxlsx, paths)

def excelpaths(paths: Iterator[FilePath]) -> Iterator[FilePath]:
    return filter(pred.isexcel, paths)

def jsonpaths(paths: Iterator[FilePath]) -> Iterator[FilePath]:
    return filter(pred.isjson, paths)

### Generates path iterators from a directory ###
def gen_pathsT_from_dir(dirpath: DirPath, T: str) -> Iterator[FilePath]:
    """Generic for path generator from directory 
    T: xlsx, excel, xls
    """
    return pathsT(fpaths(dirpath.iterdir()), T)

def gen_xlsxpaths_from_dir(dirpath: DirPath) -> Iterator[FilePath]:
    return xlsxpaths(fpaths(dirpath.iterdir()))

def gen_excelpaths_from_dir(dirpath: DirPath) -> Iterator[FilePath]:
    return excelpaths(fpaths(dirpath.iterdir()))


### sheet generator ###

def gen_sheets_from_excelpath(excelpath: FilePath) -> Iterator[str]:
    ef = pd.ExcelFile(excelpath)
    for sheet in ef.sheet_names:
        yield sheet

def gen_excelpath_sheetname_pair(excelpath: FilePath) -> Tuple[FilePath, Iterator[str]]:
    return (xlsxpath, gen_sheets_from_excelpath(excelpath))

def gen_T_sheet_pairs(dirpath: DirPath, T: str) -> Iterator[Tuple[FilePath, Iterator[str]]]:
    """Returns an iterator of (FilePath, sheet name) pairs"""
    for i in gen_xlsxpaths_from_dir(dirpath):
        yield (i, gen_sheets_from_excelpath(i))

def dict_of_file_sheetname(dirpath: DirPath) -> Dict:
    """Returns {FilePath : [sheet_name1, sheet_name2, ...]}""" 
    sheet_pairs = gen_excelpath_sheetname_pairs(dirpath)
    xlsx_dict = {}
    for pair in sheet_pairs:
        key = pair[0]
        val = list(pair[1])
        xlsx_dict[key] = val
    return xlsx_dict

def dict_from_summary_json(dirpath: DirPath) -> Dict:
    """Returns a py dict of directory contents from summary.json"""
    return json.load(open(dirpath / 'summary.json'))

def drop_non_xlsx_keys(jd: Dict) -> Dict:
    newd = {}
    for key in jd:
        if Path(key).suffix == '.xlsx':
            newd[key] = jd[key]
    return newd

def gen_summarize_excel_contents(dirpath: DirPath) -> Iterator:
    xlsx_sheet_pairs = gen_excelpath_sheetname_pairs(dirpath)
    jd = dict_from_summary_json(dirpath)
    for pair in xlsx_sheet_pairs:
        fname = pair[0]
        col_list = list(pair[1])
        description = ''
        name = str(fname.name)
        if name in jd:
            description = jd[name]           
        yield (fname, description, col_list)
    

def gen_colnames_from_sheet(xlsxpath: FilePath, sheet: str) -> Iterator[str]:
    for column in pd.read_excel(xlsxpath, sheet_name=sheet).columns:
        yield column

def gen_all_colnames(dirpath):
    xlsx_fpaths = gen_xlsx_fpaths(dirpath)
    for fpath in xlsx_fpaths:
        io = pd.ExcelFile(fpath)
        for sheet in io.sheet_names:
            yield (sheet, list(io.parse(sheet_name=sheet).columns))
    



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
   
    
