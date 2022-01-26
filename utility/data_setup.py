import pandas as pd
from pathlib import Path
from utility.meta import pipe


def sars_cov_2_ppi_to_plain_text():
    dpath = pipe("./pyext/data/sars-cov-2-ppi", Path)
    jpath = dpath / "summary.json"
    with open(str(jpath), 'r') as f:

        i = 0
        for line in f:
            pipe(f.readline(), print)


def sars_cov_2_ppi_read_excel():

    """Reads the .xlsx file and writes a plain text file """

    dpath = pipe("./pyext/data/sars-cov-2-ppi", Path)
    epath = dpath / "41586_2020_2286_MOESM5_ESM.xlsx"
    d = pd.read_excel(str(epath))
    counter=0

    write_file = dpath / "41586_2020_2286_MOESM5_ESM.csv"
    write_file = str(write_file)
    print("Warning - Data not written")
    #d.to_csv(write_file, sep="\t")


def validate_apms_data():
    dpath = pipe("./pyext/data/sars-cov-2-ppi", Path)
    dpath = dpath / "41586_2020_2286_MOESM5_ESM.csv"

    with open(str(dpath), 'r') as f:
        header = f.readline()
        headerlist = header.split("\t")
        ntabs = len(headerlist)

        baits = ["SARS-CoV2 E", "SARS-CoV2 M",  "SARS-CoV2 N",
                 "SARS-CoV2 nsp1",  "SARS-CoV2 nsp10"]
        for line in f:
            #print(line)
            linelist = line.split("\t")
            assert len(linelist) == ntabs
            assert linelist[1] in baits
