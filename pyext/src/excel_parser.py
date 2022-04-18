import pandas as pd
from pathlib import Path

from .typedefs import AnyPath, DirPath, FilePath


def write_excel_files(working_dir: DirPath) -> list[FilePath]:
    for file in working_dir.iterdir():
        file = Path(file)
        if file.is_file():
            if file.suffix in (".xls", ".xlsx"):
                file = str(file)
                xlsx_promt(file)


def xlsx_promt(file):
    print(f"view file? {file} y/n")
    i = input()
    if i in ("y", "Y"):
        parse_sheets(file)


def parse_sheets(file):
    xl = pd.ExcelFile(file)
    sheets = xl.sheet_names
    print(sheets)
    print("export sheets?")
    i = input()
    if i in ("y", "Y"):
        for sheet in sheets:
            export_sheet(sheet, file)


def export_sheet(sheet, file):
    cases = ("tsv", "csv", "psv", "skip", "other")
    print(f"format? {cases}")
    i = input()
    while i not in cases:
        print(f"invalid: try {cases}")
        i = input()

    for mcase in cases:
        if mcase in ("tsv", "csv", "psv", "other"):
            _export_sheet(file, sheet, mcase)


def _export_sheet(file, sheet, mcase):
    write_name = format_filename(file, sheet, mcase)
    d = pd.read_excel(file, sheet_name=sheet)
    sep = get_sep(mcase)
    print(f"write file? {write_name}")
    i = input()
    if i == "y":
        print(f"-> {write_name}")
        d.to_csv(write_name, sep=sep)

    else:
        print("Aborted")


def get_sep(mcase: str) -> str:
    if sep == "tsv":
        return "\t"
    elif sep == "csv":
        return ","
    elif sep == "psv":
        return "|"
    elif sep == "other":
        print("type seperator character")
        i = input()
        return i
    else:
        print("Fallthrough error")
        assert False


def concat(sl: list[str]) -> str:
    s = ""
    for i in sl:
        s += i
    return s


def format_filename(file, sheet, mcase):
    write_name = file.strip(".xls")
    write_name = write_name.strip(".xlsx")
    write_name = write_name.split(" ")
    write_name = concat(write_name)

    sheet = sheet.split(" ")
    sheet = concat(sheet)
    write_name += sheet + "." + mcase
    return write_name
