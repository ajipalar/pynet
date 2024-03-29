from collections.abc import Generator
from pyext.src.typedefs import AnyCol as AnyCol, AnyPath as AnyPath, Bait as Bait, DataFrame as DataFrame, DirPath as DirPath, FilePath as FilePath, Organism as Organism, PGGroupCol as PGGroupCol, PreyUID as PreyUID, UID as UID
from typing import Any, Dict, Iterator, List, Set, Tuple

def pathsT(paths: Iterator[FilePath], T: str) -> Iterator[FilePath]: ...
def fpaths(paths: Iterator[AnyPath]) -> Iterator[FilePath]: ...
def xlsxpaths(paths: Iterator[FilePath]) -> Iterator[FilePath]: ...
def excelpaths(paths: Iterator[FilePath]) -> Iterator[FilePath]: ...
def jsonpaths(paths: Iterator[FilePath]) -> Iterator[FilePath]: ...
def gen_pathsT_from_dir(dirpath: DirPath, T: str) -> Iterator[FilePath]: ...
def gen_xlsxpaths_from_dir(dirpath: DirPath) -> Iterator[FilePath]: ...
def gen_excelpaths_from_dir(dirpath: DirPath) -> Iterator[FilePath]: ...
def gen_sheets_from_excelpath(excelpath: FilePath) -> Iterator[str]: ...
def excelpath_sheetname_pair(excelpath: FilePath) -> Tuple[FilePath, Iterator[str]]: ...
def gen_excelpath_sheetname_pairs(dirpath: DirPath) -> Iterator[Tuple[FilePath, Iterator[str]]]: ...
def dict_of_file_sheetname(dirpath: DirPath) -> Dict: ...
def dict_from_summary_json(dirpath: DirPath) -> Dict: ...
def drop_non_excel_keys(jd: Dict) -> Dict: ...
def gen_summarize_excel_contents(dirpath: DirPath) -> Iterator: ...
def df_excel_summary(dirpath: DirPath) -> DataFrame: ...
def gen_colnames_from_sheet(xlsxpath: FilePath, sheet: str) -> Iterator[str]: ...
def gen_all_colnames(dirpath) -> Generator[Any, None, None]: ...
def tree_excel(p) -> None: ...
def condf(f, condition, *args): ...
def treef(p, f=..., *args) -> None: ...
def print_all_colnames(path) -> None: ...
def gen_xls(xlsfile: FilePath) -> Iterator[str]: ...
def parse_lip_line(line: str) -> List[str]: ...
def gen_parsed_lip_xls(xls: FilePath) -> Iterator[List[str]]: ...
def filter_lines(line_list: List[str], column_positions: List[int]) -> List[str]: ...
def gen_filter_xls_columns(it: Iterator[List[str]], column_positions: List[int]) -> Iterator[List[str]]: ...
def desired_columns() -> List[str]: ...
def build_position_dict(sl: List[str]) -> Dict: ...
def check_desired_columns_in_header(header_cols: List[str], desired_cols: List[str]): ...
def desired_positions(input_header_list: List[str], columns: List[str]) -> List[int]: ...
def parse_lip_xls_file(xlspath: FilePath, cols: List[AnyCol] = ...) -> Iterator[List[AnyCol]]: ...
def gen_helper_protein_name_columns(xlspath: FilePath, cols=...) -> Iterator[List[PGGroupCol]]: ...
def gen_parse_protein_names_from_it(col_it: Iterator[List[PGGroupCol]]) -> Iterator[List[str]]: ...
def unique_protein_names(protein_name_it: Iterator[List[str]]) -> Tuple[PGGroupCol, Set[UID]]: ...
def wrapper_pname_set(x: FilePath) -> Tuple[PGGroupCol, Set[UID]]: ...
def df_from_it(xls_it: Iterator[List[str]]): ...
def gen_xls_column_stream(f: FilePath, col: str) -> Iterator[str]: ...
def get_header(xlsfile: FilePath) -> List[str]: ...
def remove_irrelevant_columns(it: Iterator[str]) -> Iterator[str]: ...
def xls_shape(xlsfile: FilePath) -> Tuple[int, int]: ...
def load_lip_datasets(lipdirpath: DirPath, prints: bool = ...) -> Iterator[Tuple[FilePath, Set[UID]]]: ...
def load_gordon_dataset(apmsxlsx: FilePath) -> Dict[Bait, Set[PreyUID]]: ...
def load_stuk_dataset(stukpath: FilePath) -> Dict[Organism, Dict[Bait, Set[PreyUID]]]: ...
def check_prey(prey) -> None: ...
def check_gordon(gordon: Dict[Bait, Set[PreyUID]]): ...
def check_stuk(sdict: Dict[Organism, Dict[Bait, Set[PreyUID]]]): ...
def check_lip_tuple(fp, pset) -> None: ...
def check_lip(liplist: List[Tuple[FilePath, Set[UID]]]): ...
def do_comparison(stukpath: FilePath, gordonpath: FilePath, lippath: DirPath, prints: bool = ...): ...
