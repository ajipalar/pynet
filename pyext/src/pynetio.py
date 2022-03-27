from . import predicates as pred
from .typedefs import (
    AnyPath,
    AnyCol,
    Bait,
    DataFrame,
    DirPath,
    FilePath,
    Organism,
    PGGroupCol,
    PreyUID,
    UID
)

import argparse
import json
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import typing
from typing import Dict, Iterable, Iterator, List, Set, Tuple

"""
Reading and parsing excel files
pf pure function. Function has no side effects
"""

#parser = argparse.ArgumentParser()
#parser.add_argument("-d", help="Read all excel files in the directory")

#args = parser.parse_args()

# Filters paths using predicates #
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


# Generates path iterators from a directory
def gen_pathsT_from_dir(dirpath: DirPath, T: str) -> Iterator[FilePath]:
    """Generic for path generator from directory
    T: xlsx, excel, xls
    """
    return pathsT(fpaths(dirpath.iterdir()), T)


def gen_xlsxpaths_from_dir(dirpath: DirPath) -> Iterator[FilePath]:
    return xlsxpaths(fpaths(dirpath.iterdir()))


def gen_excelpaths_from_dir(dirpath: DirPath) -> Iterator[FilePath]:
    return excelpaths(fpaths(dirpath.iterdir()))


# sheet generator ###

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
        df = pd.ExcelFile(fpath)
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


def treef(p, f=lambda pathy: None, *args):
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


# Read .xls files using iterators instead of pandas ###
def gen_xls(xlsfile: FilePath) -> Iterator[str]:
    with open(xlsfile, 'r') as f:
        for line in f:
            yield line


def parse_lip_line(line: str) -> List[str]:
    return list(i.strip() for i in line.split('\t'))


def gen_parsed_lip_xls(xls: FilePath) -> Iterator[List[str]]:
    lip_gen = gen_xls(xls)
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
               'PEP.IsProteinGroupSpecific', 'PEP.IsProteotypic', 
               'PEP.NrOfMissedCleavages', 'PEP.PeptidePosition', 
               'PEP.StrippedSequence', 'PEP.DigestType - [Trypsin/P]',
               'EG.iRTPredicted', 'EG.ModifiedPeptide', 'EG.ModifiedSequence', 'EG.Qvalue',
               'FG.Charge', 'FG.LabeledSequence', 'FG.PrecMz', 'FG.MS2RawQuantity']
    return columns


def build_position_dict(sl: List[str]) -> Dict:
    d = {}
    for i, k in enumerate(sl):
        d[k] = i
    return d


def check_desired_columns_in_header(header_cols: List[str], desired_cols: List[str]):
    for col in desired_cols:
        try:
            assert col in header_cols
        except AssertionError:
            raise AssertionError(f'{col} not in header:\n\n{header_cols}')


def desired_positions(input_header_list: List[str], columns: List[str]) -> List[int]:
    pos_dict = build_position_dict(input_header_list)
    pos_list = []
    for i in columns:
        pos_list.append(pos_dict[i])
    return pos_list


def parse_lip_xls_file(xlspath: FilePath,
                       cols: List[AnyCol] = desired_columns()
                       ) -> Iterator[List[AnyCol]]:
    # Get the desired columns

    # Create a file Iterator[List[str]] where the str are whitespace stripped
    xls_it = gen_parsed_lip_xls(xlspath)
    header_list = next(xls_it)
    #Make sure the desired columns are in the header_list
    check_desired_columns_in_header(header_list, cols)
    #Get the column positions
    pos_list = desired_positions(header_list, cols)
    filtered_header = filter_lines(header_list, pos_list)
    #generate the filtered file
    yield filtered_header
    for line in xls_it:
        yield filter_lines(line, pos_list)


def gen_helper_protein_name_columns(xlspath: FilePath, cols=['PG.ProteinGroups']) -> Iterator[List[PGGroupCol]]:
    return parse_lip_xls_file(xlspath, cols=cols)


def gen_parse_protein_names_from_it(col_it: Iterator[List[PGGroupCol]]) -> Iterator[List[str]]:
    #Yeild the header
    yield next(col_it)
    for name_list in col_it:
        semicolon_str = name_list[0]
        protein_id_list = semicolon_str.split(";")
        yield protein_id_list



def unique_protein_names(protein_name_it: Iterator[List[str]]) -> Tuple[PGGroupCol, Set[UID]]:
    header_str = next(protein_name_it)[0]
    s = set()
    for i, protein_id_list in enumerate(protein_name_it):
        for protein_name in protein_id_list:
            s.add(protein_name)
    return header_str, s



def wrapper_pname_set(x: FilePath) -> Tuple[PGGroupCol, Set[UID]]:
    f = gen_helper_protein_name_columns
    g = gen_parse_protein_names_from_it
    h =  unique_protein_names
    return h(g(f(x)))


"""
def get_intersections(dirpath: DirPath):
    excel_paths = excelpaths(dirpath)
    n = len(excel_paths)
    intersecrtion_matrix = np.ndarray((n, n), dtype=int)
    columns = []
    for fpath in  excel_paths:
"""

def df_from_it(xls_it: Iterator[List[str]]):
    columns = next(xls_it)
    d = pd.DataFrame(columns=columns)

    in_development_col_list = ['R.Condition', 'R.FileName', 'R.Label',
        'R.Replicate', 'PG.ProteinAccessions', 'PG.ProteinGroups',
        'PG.Qvalue', 'PG.Quantity', 'PEP.IsProteinGroupSpecific',
        'PEP.IsProteotypic', 'PEP.NrOfMissedCleavages', 'PEP.PeptidePosition',
        'PEP.StrippedSequence', 'PEP.DigestType - [Trypsin/P]',
        'EG.iRTPredicted', 'EG.ModifiedPeptide', 'EG.ModifiedSequence',
        'EG.Qvalue', 'FG.Charge', 'FG.LabeledSequence',
        'FG.PrecMz', 'FG.MS2RawQuantity']

    dtypes = {'R.Condition': "category", 'R.Replicate': 'category',
        'PEP.DigestType - [Trypsin/P]': "category", }
    for i, lst in enumerate(xls_it):
        d.loc[len(d.index)] = lst
    return d


def gen_xls_column_stream(f: FilePath, col: str) -> Iterator[str]:
    #Unparsed lip file iterator
    xls_it = gen_parsed_lip_xls(xls)
    header_list = next(xls_it)


def get_header(xlsfile: FilePath) -> List[str]:
    with open(xlsfile, 'r') as f:
        header = f.readline()
        header = header.split('\t')
        header = list(i.strip() for i in header)
        return header


def remove_irrelevant_columns(it: Iterator[str]) -> Iterator[str]:

    headers = ['R.Condition', 'R.FileName', 'R.Fraction', 'R.Label', 
       'R.Replicate', 'PG.ProteinAccessions', 'PG.ProteinGroups', 'PG.Cscore',
       'PG.Pvalue', 'PG.Qvalue', 'PG.RunEvidenceCount', 'PG.Quantity',
       'PEP.AllOccurringProteinAccessions', 'PEP.GroupingKey', 
       'PEP.GroupingKeyType', 'PEP.IsProteinGroupSpecific',
       'PEP.IsProteotypic', 'PEP.NrOfMissedCleavages', 
       'PEP.PeptidePosition', 'PEP.StrippedSequence',
       'PEP.DigestType - [Trypsin/P]', 'PEP.Rank', 
       'PEP.RunEvidenceCount', 'PEP.UsedForProteinGroupQuantity',
       'EG.IntPIMID', 'EG.iRTPredicted', 'EG.IsDecoy', 
       'EG.ModifiedPeptide', 'EG.ModifiedSequence', 'EG.UserGroup', 
       'EG.Workflow', 'EG.IsUserPeak', 'EG.IsVerified', 'EG.Qvalue', 
       'EG.ApexRT', 'EG.iRTEmpirical', 'EG.RTPredicted',
       'EG.AvgProfileQvalue', 'EG.MaxProfileQvalue', 
       'EG.MinProfileQvalue', 'EG.PercentileQvalue', 
       'EG.ReferenceQuantity (Settings)', 
       'EG.TargetQuantity (Settings)', 
       'EG.TotalQuantity (Settings)', 
       'EG.UsedForPeptideQuantity', 
       'EG.UsedForProteinGroupQuantity', 
       'EG.Cscore', 'FG.Charge', 'FG.IntMID', 'FG.LabeledSequence', 
       'FG.PrecMz', 'FG.PrecMzCalibrated', 'FG.MS2Quantity', 
       'FG.MS2RawQuantity', 'FG.Quantity']

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

#Load in the lip Datasets
def load_lip_datasets(lipdirpath: DirPath,
                      prints=False
                      ) -> Iterator[Tuple[FilePath, Set[UID]]]:

    lip_xls_paths = excelpaths(lipdirpath.iterdir())
    def gen_protein_sets(x: Iterator[FilePath],
                         prints=prints) -> Iterator[Tuple[FilePath, Set[UID]]]:
        for path in x:
            if prints: print(f'wrapping {path}')
            header, pset = wrapper_pname_set(path)
            yield (path.name, pset)
    return gen_protein_sets(lip_xls_paths)


#Load in the Gordon APMS Dataset
def load_gordon_dataset(apmsxlsx: FilePath) -> Dict[Bait, Set[PreyUID]]:
    d = pd.read_excel(apmsxlsx)
    unique_baits = set(d['Bait'])
    bait_preys_dict= {key: set() for key in unique_baits}
    for i, row in d.iterrows():
        key: Bait = row['Bait']
        val: PreyUID = row['Preys']
        bait_preys_dict[key].add(val)
    return bait_preys_dict


#Load in the Stuk dataset
def load_stuk_dataset(stukpath: FilePath
                      ) -> Dict[Organism, Dict[Bait, Set[PreyUID]]]:
    stuk_significant_apms = pd.read_excel(stukpath,
                                          usecols=['bait_organism',
                                                   'bait_name',
                                                   'majority_protein_acs'],
                                          sheet_name='A - Significant interactions')

    unique_organisms = set(stuk_significant_apms.iloc[:, 0])
    unique_baits = set(stuk_significant_apms.iloc[:, 1])
    bait_dict = {bait: list() for bait in unique_baits}
    #Dict[org, Dict[bait, preys]

    stuk_preys_dict = {org: {bait: set() for bait in unique_baits} for org in unique_organisms}
    #populate the dictionary with unique preys
    for i, row in stuk_significant_apms.iterrows():
        org = row['bait_organism']
        bait = row['bait_name']
        uid_list = row['majority_protein_acs'].split(';')
        #Remove isoforms
        uid_list = list(i.split('-')[0] for i in uid_list)
        uid_list = list(i.split('#')[0] for i in uid_list)

        for uid in uid_list:
            stuk_preys_dict[org][bait].add(uid)

    #Remove empty sets from the dictionary
    to_remove = []
    for org in stuk_preys_dict:
        for bait in stuk_preys_dict[org]:
            if len(stuk_preys_dict[org][bait]) == 0:
                to_remove.append((org, bait))

    for org, bait in to_remove:
        stuk_preys_dict[org].pop(bait)

    return stuk_preys_dict


def check_prey(prey):
    assert prey.isalnum()
    assert prey.isupper()
    try:
        assert len(prey) < 11
    except AssertionError:
        raise AssertionError(f'{prey, len(prey)}')

def check_gordon(gordon: Dict[Bait, Set[PreyUID]]):
    for bait, preylist in gordon.items():
        for prey in preylist:
            check_prey(prey)

def check_stuk(sdict: Dict[Organism, Dict[Bait, Set[PreyUID]]]):
    for org in sdict:
        for bait in sdict[org]:
            for prey in sdict[org][bait]:
                try:
                    check_prey(prey)
                except AssertionError:
                    raise AssertionError(f'{prey}')

def check_lip_tuple(fp, pset):
    for uid in pset:
        try:
            check_prey(uid)
        except AssertionError:
            if uid == 'NSP7_SARS2' or uid == 'NSP9_SARS2':
                print(fp, uid)
                continue
            else:
                raise AssertionError(f'{uid}')

def check_lip(liplist: List[Tuple[FilePath, Set[UID]]]):
    for fp, pset in liplist:
        check_lip_tuple(fp, pset)

def do_comparison(stukpath: FilePath,
                  gordonpath: FilePath,
                  lippath: DirPath,
                  prints=False):
    gordon_data:     Dict[Bait, set[PreyUID]]
    stuk_data:       Dict[Organism, dict[Bait, set[PreyUID]]]
    lip_data:        List[Tuple[FilePath, set[UID]]]

    if prints: print(f'Done\nLoading Gillet data')
    lip_data = list(load_lip_datasets(lippath, prints=prints))
    check_lip(lip_data)
    if prints: print(f'Passed')

    if prints: print(f'Loading Stukalov data')
    stuk_data = load_stuk_dataset(stukpath)
    if prints: print(f'Done\nLoading Gordon data')
    gordon_data = load_gordon_dataset(gordonpath)
    if prints: print(f'Passed\nChecking Gordon data')
    check_gordon(gordon_data)
    if prints: print(f'Done\nChecking Stukalov data')
    check_stuk(stuk_data)
    if prints: print(f'Passed\nChecking Gillet data')

    def update_gordon_keys(gd):
        newd = {}
        for key, pset in gd.items():
            newd[f'Krogan_{key}'] = pset
        return newd

    krogan_dict = update_gordon_keys(gordon_data)

    def update_stuk_dict(sd):
        newd = {}
        for org in sd:
            for bait in sd[org]:
                newd[f'Stuk_{org}_{bait}'] = sd[org][bait]
        return newd

    stuk_dict = update_stuk_dict(stuk_data)

    def update_lip_data(ld):
        """Parse the lip filename to a unique key"""

        def make_lip_key(fp):
            keylist = fp.split("_")[2:]
            s=''
            for k in keylist:
                s += k 
            return s.strip('_SpecLib_Report.xls')

        lip_dict = {}
        for i, t in enumerate(ld):
            s, pset = t[0], t[1]
            s = make_lip_key(s)
            lip_dict[s] = pset
        return lip_dict

    lip_dict = update_lip_data(lip_data)

    lip_dict = lip_dict | stuk_dict
    del stuk_dict
    lip_dict = lip_dict | krogan_dict

    def intersection_matrix(d):
        n = len(d)
        M = np.ndarray((n,n), dtype=int)
        keys = list(d.keys())
        keyl = len(keys)
        for i, key1 in enumerate(keys):
            for j in range(i, keyl):
                key2 = keys[j]
                M[i,j]=len(d[key1].intersection(d[key2]))
        return pd.DataFrame(M, columns=keys, index=keys)
 
    return intersection_matrix(lip_dict)

# Load in the BioID dataset

# Load in the genetic screen

#if args.d:
#   print_all_colnames(args.d)
