import pandas as pd
from pathlib import Path
print('What file would you like to convert?')
wd = Path(".")

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


for file in wd.iterdir():
    if file.is_file():
        print(file)
        answer = input()
        if answer == 'y':
            convert_file()


