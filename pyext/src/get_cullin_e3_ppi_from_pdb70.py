"""
Run this script from the project home directory as
python -m pyext.src.data.get_cullin_e3_from_pdb70
"""
import re
import click
import numpy as np
import pandas
import subprocess
from enum import Enum
from pathlib import Path

from cullin_benchmark_test import (
    CullinBenchMark,
    accumulate_indicies,
    bar_plot_df_summary,
    binary_search,
    biogrid_df_report,
    check_biogrid_data,
    check_bounds,
    compare_reports,
    find_bounds,
    find_end,
    find_start,
    format_biogrid_df_report,
    get_all_indicies,
    get_biogrid_summary,
    get_json_report_from_report,
    make_bounds,
    show_idmapping_results,
    transform_and_validate_biogrid,
    uniprot_id_mapping,
)
import HHBlits

def step4(prob_threshold: float, dev_path=False):
    """
    Count the pairwise interactions among the benchmark set, writing these data to an interm csv
    Only consider pdb_ids from hhblits .hhr (.log) files above a certain probability threshold

    Params:
      prob_threshold: float
    """

    # First we consider N proteins for an N x N matrix 

    path = str(PyPath.interim.value)
    path = Path(path)

    path = path / "cullin_e3_ligase/hhblits_out/"

    if dev_path:
        path = Path("../../data/interim/cullin_e3_ligase/hhblits_out")
    uniprot_ids = []

    N = 0
    for p in path.iterdir():
        if p.is_file() and p.suffix == ".log":
            N+=1
    assert 2000 < N < 3000
    
    # {pdb_id: uids} 
    pdb_uid = {}

    for i, p in enumerate(path.iterdir()):
        if i%300 == 0:
            print(i)
        if p.is_file() and p.suffix == ".log":
            hhr_data: dict = HHBlits.parse_log(str(p))
            hhr_data: HHBLits.HHResults = HHBlits.HHResults(**hhr_data)
            hhr_data.to_numeric()  # change the string fields to python ints and floats
            hhr_data.validate()
            assert isinstance(hhr_data.uid, str)
            uniprot_ids.append(hhr_data.uid)

            float_arr = np.array([j / 100 for j in hhr_data.prob])
            index = np.where(float_arr > prob_threshold)
            float_arr = float_arr[index]

            end = len(float_arr)

            hhr_data.select(0, end)  # select the top pdb_ids above prob threshold

            for pdb_id in hhr_data.pdb_id:
                if pdb_id not in pdb_uid:
                    pdb_uid[pdb_id] = [hhr_data.uid]
                else:
                    pdb_uid[pdb_id].append(hhr_data.uid)

    assert len(uniprot_ids) == N
    return pdb_uid



@click.command()
@click.option("--validate-project/--no-validate-project", default=True, help='validate the directory structure')
@click.option("--get-cullin-ids/--no-get-cullin-ids", default=False, help='get cullin Uniprot Ids')
def main(validate_project,
         get_cullin_ids):
    #0 Do some checking

    def step0():
        for p in [member.value for member in PyPath]:
            assert p.is_dir()
    
    def step1():
        """
        Get the list of cullin UniProt Accession ID's or GIDs
        """
        cullin_benchmark = CullinBenchMark(dirpath=PyPath.cullin_e3_ligase.value)
        click.echo(cullin_benchmark)
        click.echo(cullin_benchmark.baits)
        click.echo(cullin_benchmark.data.columns)

        prey = list(set(cullin_benchmark.data["Prey"]))
        bait = list(set(cullin_benchmark.data["Bait"]))

        click.echo(bait)
#        click.echo(prey)
        return bait, prey
    
    def step2(cullin_benchmark):
        """
        Get a representative sequence for each of the ids including viral sequences

        The sequences were obtained from uniprot by keyword search

        """

        #UP-ID
        keywords = {"CUL5": "Q93034",
                    "CBFB": "Q13951",
                    "ELOB": "Q15370",
                    "vifprotein" : "P69723"}
                
        # All other baits are human with a uniprot ID

        # The keys are the "Prey" keys used in the study
        # These are Uniprot ID's and strings the authors defined
        # that correspond primarily to viral proteins
        # https://www.sciencedirect.com/science/article/pii/S1931312819302537?via%3Dihub#bib45
        # The values are the primary amino acid sequence of each
        # protein as defined by uniprot
        # All HIV translational products are included for completness 

        seq_db = {"vifprotein": "MENRWQVMIVWQVDRMRIRTWKSLVKHHMYVSGKARGWFYRHHYESPHPRISSEVHIPLGDARLVITTYWGLHTGERDWHLGQGVSIEWRKKRYSTQVDPELADQLIHLYYFDCFSDSAIRKALLGHIVSPRCEYQAGHNKVGSLQYLALAALITPKKIKPPLPSVTKLTEDRWNKPQKTKGHRGSHTMNGH",
                  "polyprotein": "",
                  "nefprotein": "",
                  "tatprotein": "",
                  "gagpolyprotein": "",
                  "revprotein": "",
                  "IGHG1_MOUSE": "",
                  "envpolyprotein": "",
                  "CBFB": "MPRVVPDQRSKFENEEFFRKLSRECEIKYTGFRDRPHEERQARFQNACRDGRSEIAFVATGTNLSLQFFPASWQGEQRQTPSREYVDLEREAGKVYLKAPMILNGVCVIWKGWIDLQRLDGMGCLEFDEERAQQEDALAQQAFEEARRRTREFEDRDRSHREEMEVRVSQLLAVTGKKTTRP",
                  "CUL5": "MATSNLLKNKGSLQFEDKWDFMRPIVLKLLRQESVTKQQWFDLFSDVHAVCLWDDKGPAKIHQALKEDILEFIKQAQARVLSHQDDTALLKAYIVEWRKFFTQCDILPKPFCQLEITLMGKQGSNKKSNVEDSIVRKLMLDTWNESIFSNIKNRLQDSAMKLVHAERLGEAFDSQLVIGVRESYVNLCSNPEDKLQIYRDNFEKAYLDSTERFYRTQAPSYLQQNGVQNYMKYADAKLKEEEKRALRYLETRRECNSVEALMECCVNALVTSFKETILAECQGMIKRNETEKLHLMFSLMDKVPNGIEPMLKDLEEHIISAGLADMVAAAETITTDSEKYVEQLLTLFNRFSKLVKEAFQDDPRFLTARDKAYKAVVNDATIFKLELPLKQKGVGLKTQPESKCPELLANYCDMLLRKTPLSKKLTSEEIEAKLKEVLLVLKYVQNKDVFMRYHKAHLTRRLILDISADSEIEENMVEWLREVGMPADYVNKLARMFQDIKVSEDLNQAFKEMHKNNKLALPADSVNIKILNAGAWSRSSEKVFVSLPTELEDLIPEVEEFYKKNHSGRKLHWHHLMSNGIITFKNEVGQYDLEVTTFQLAVLFAWNQRPREKISFENLKLATELPDAELRRTLWSLVAFPKLKRQVLLYEPQVNSPKDFTEGTLFSVNQEFSLIKNAKVQKRGKINLIGRLQLTTERMREEENEGIVQLRILRTQEAIIQIMKMRKKISNAQLQTELVEILKNMFLPQKKMIKEQIEWLIEHKYIRRDESDINTFIYMA",
                  "ELOB": "MDVFLMIRRHKTTIFTDAKESSTVFELKRIVEGILKRPPDEQRLYKDDQLLDDGKTLGECGFTSQTARPQAPATVGLAFRADDTFEALCIEPFSSPPELPDVMKPQDSGSSANEQAVQ"
                  }
        
        prey = list(set(cullin_benchmark.data["Prey"]))

        # 21 ids failed to map
        # 8368 protein sequences were found
        # 


    def step3():
        """
        Query pdb70 for PDB files with a sequence within threshold, writing these to an interim directory
        """
        pdb_dict = {}
        preyids = [] # populate this list

        for uid in uids:
            pdbs = []
            # find pdbs
            for pdb in pdbs:
                if pdb not in pdb_dict:
                    pdb_dict[pdb] = [uid]
                else:
                    l = pdb_dict[pdb]
                    l.append(uid)
                    pdb_dict[pdb] = l

        return pdb_dict
    
    if validate_project:
        step0()

    #1 Get the list of cullin UniProt Accession ID's or GIDs
    if get_cullin_ids:
        step1()

    #2 Get a representative sequence for each of the ids including viral sequences

    #3 Query pdb70 for PDB files with a sequence within threshold, writing these files to an interm directory

    #4 Count the pairwise interactions among the benchmark set, writing these data to an interim csv  

class PyPath(Enum):
    """
    Global immutable path variables
    """

    home = Path(".")
    pyext = home / "pyext"
    src = pyext / "src"
    data = home / "data"
    raw = data / "raw"
    interim = data / "interim"
    cullin_e3_ligase = raw / "cullin_e3_ligase"

is_aa = re.compile("^M[ACDEFGHIKLMNPQRSTVWY]*")

def check_sequence(aa: str):
    """
    check a primary amino acid sequence for common assumptions
    """

    assert len(aa) > 10
    assert is_aa.match(aa)



if __name__ == "__main__":
    main()
