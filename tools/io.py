import argparse
import json
from pathlib import Path
import pandas as pd
import sys
import typing as t
from typing import Dict, Iterable, Iterator, List, Tuple

#Custom modules
import tools.predicates as pred
from net.typedefs import AnyPath, FilePath, DataFrame, DirPath


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

def excelpath_sheetname_pair(excelpath: FilePath) -> Tuple[FilePath, Iterator[str]]:
    return (excelpath, gen_sheets_from_excelpath(excelpath))

def gen_excelpath_sheetname_pairs(dirpath: DirPath) -> Iterator[Tuple[FilePath, Iterator[str]]]:
    return (excelpath_sheetname_pair(i) for i in excelpaths(fpaths(dirpath.iterdir())))

def dict_of_file_sheetname(dirpath: DirPath) -> Dict:
    """Returns {FilePath : [sheet_name1, sheet_name2, ...]}""" 
    sheet_pairs = gen_excelpath_sheetname_pairs(dirpath)
    excel_dict = {}
    for pair in sheet_pairs:
        key = pair[0]
        val = list(pair[1])
        excel_dict[key] = val
    return excel_dict

def dict_from_summary_json(dirpath: DirPath) -> Dict:
    """Returns a py dict of directory contents from summary.json"""
    return json.load(open(dirpath / 'summary.json'))

def drop_non_excel_keys(jd: Dict) -> Dict:
    newd = {}
    for key in jd:
        if pred.isexcel(Path(key)):
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

def df_excel_summary(dirpath: DirPath) -> DataFrame:    
    gen = gen_summarize_excel_contents(dirpath)
    colnames = ['Name', 'Description', 'sheetnames', 'Fullpath'] 
    gen = ((str(i[0].name), i[1], i[2], str(i[0])) for i in gen)
    return pd.DataFrame(list(gen), columns=colnames)
    

def gen_colnames_from_sheet(xlsxpath: FilePath, sheet: str) -> Iterator[str]:
    for column in pd.read_excel(xlsxpath, sheet_name=sheet).columns:
        yield column

def gen_all_colnames(dirpath):
    xlsx_fpaths = gen_xlsx_fpaths(dirpath)
    for fpath in xlsx_fpaths:
        df  = pd.ExcelFile(fpath)
        for sheet in df.sheet_names:
            yield (sheet, list(df.parse(sheet_name=sheet).columns))
    



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

### Read .xls files using iterators instead of pandas ###
def gen_xls(xlsfile: FilePath) -> Iterator[str]:
    with open(xlsfile, 'r') as f:
        for line in f:
            yield f.readline()

def parse_lip_line(line: str) List[str]:
    return list(i.strip() for i in line.split('\t'))

def gen_parsed_lip_xls(xlsfile: FilePath) -> Iterator[List[str]]:
    lip_gen = gen_xls(xlsfile)
    return (parse_lip_line(i) for i in lip_gen)

def filter_lines(line_list: List[str], column_positions: List[int]) -> List[str]:
    new_list = []
    for pos in column_positions:
        new_list.append(line_list[pos])
    return new_list
    
def gen_filter_xls_columns(it: Iterator[List[str]], column_positions: List[int]) -> Iterator[List[str]]:
    for stripped_str_list in it:
        yield filter_lines(stripped_str_list, column_positions)

def desired_columns() -> List[str]:
    columns = ['R.Condition', 'R.FileName', 'R.Label', 'R.Replicate', 
               'PG.ProteinAccessions', 'PG.ProteinGroups',  'PG.Qvalue', 'PG.Quantity', 
               'PEP.IsProteinGroupSpecific', 'PEP.IsProteotypic', 'PEP.NrOfMissedCleavages', 'PEP.PeptidePosition', 'PEP.StrippedSequence', 'PEP.DigestType - [Trypsin/P]',   
               'EG.iRTPredicted', 'EG.ModifiedPeptide', 'EG.ModifiedSequence', 'EG.Qvalue', 
               'FG.Charge', 'FG.LabeledSequence', 'FG.PrecMz', 'FG.MS2RawQuantity'] 
    return columns

def build_position_dict(l: List[str]) -> Dict:
    d = {}
    for i, k in enumerate(l):
        d[k] = i
    return d

def check_desired_columns_in_header(header_cols: List[str], desired_cols: List[str]):
    for col in desired_cols:
        assert col in header_cols 

def desired_positions(input_header_list: List[str], columns: List[str]) -> List[int]:
    



def get_header(xlsfile: FilePath) -> List[str]:
    with open(xlsfile, 'r') as f:
        header = f.readline()
        header = header.split('\t')
        header = list(i.strip() for i in header)
        return header


def remove_irrelevant_columns(it: Iterator[str]) -> Iterator[str]:
    headers = ['R.Condition', 'R.FileName', 'R.Fraction', 'R.Label', 'R.Replicate', 
               'PG.ProteinAccessions', 'PG.ProteinGroups', 'PG.Cscore', 'PG.Pvalue', 'PG.Qvalue', 'PG.RunEvidenceCount', 'PG.Quantity', 
               'PEP.AllOccurringProteinAccessions', 'PEP.GroupingKey', 'PEP.GroupingKeyType', 'PEP.IsProteinGroupSpecific', 
               'PEP.IsProteotypic', 'PEP.NrOfMissedCleavages', 'PEP.PeptidePosition', 'PEP.StrippedSequence', 
               'PEP.DigestType - [Trypsin/P]', 'PEP.Rank', 'PEP.RunEvidenceCount', 'PEP.UsedForProteinGroupQuantity', 
               'EG.IntPIMID', 'EG.iRTPredicted', 'EG.IsDecoy', 'EG.ModifiedPeptide', 'EG.ModifiedSequence', 'EG.UserGroup', 'EG.Workflow', 'EG.IsUserPeak', 'EG.IsVerified', 'EG.Qvalue', 'EG.ApexRT', 'EG.iRTEmpirical', 'EG.RTPredicted', 
               'EG.AvgProfileQvalue', 'EG.MaxProfileQvalue', 'EG.MinProfileQvalue', 'EG.PercentileQvalue', 'EG.ReferenceQuantity (Settings)', 
               'EG.TargetQuantity (Settings)', 'EG.TotalQuantity (Settings)', 'EG.UsedForPeptideQuantity', 'EG.UsedForProteinGroupQuantity', 'EG.Cscore', 
               'FG.Charge', 'FG.IntMID', 'FG.LabeledSequence', 'FG.PrecMz', 'FG.PrecMzCalibrated', 'FG.MS2Quantity', 'FG.MS2RawQuantity', 'FG.Quantity']
    positions = build_position_dict(headers)
    header = list(next(it)).split('\t')
    
    assert len(header) == 55


def xls_shape(xlsfile: FilePath) -> Tuple[int, int]:
    r = 0
    c = 0
    with open(xlsfile, 'r') as f:
        c = len(f.readline().split('\t'))
        for line in f:
            r+=1
    return (r, c)
        
    



if args.d:
   print_all_colnames(args.d)
   
    
