import pandas as pd
from pathlib import Path

def convert_file(file):
    xl = pd.ExcelFile(file)
    for j , sheet in enumerate(xl.sheet_names):
        sheetl = sheet.split(' ')
        s=""
        for term in sheetl:
            s+=term + "_"
        s = s.strip("_")
        write_path = str(file).strip('xls') + s + '.csv'
        d = pd.read_excel(file, sheet_name = j) 
        d.to_csv(write_path)


def query_user():
    wd = Path(".")

    for file in wd.iterdir():
        if file.is_file():
            print(f"convert {file}? [y/n]")
            answer = input()
            if answer == 'y':
                convert_file(file)

if __name__ == "__main__":
    query_user()
