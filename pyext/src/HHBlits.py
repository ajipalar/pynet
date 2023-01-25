import re
import pandas as pd
import json
from dataclasses import dataclass, KW_ONLY
from collections import namedtuple

@dataclass
class HHResults:
    _: KW_ONLY
    uid: str
    match_columns: int
    no_of_seqs: str
    neff: int
    searched_hmms: int
    date: str
    command: str

    pdb_id: str
    chain_id: str
    description: str
    note: list
    prob: list
    e_value: list
    p_value: list
    score: list
    cols: list
    query_start: list
    query_end: list
    template_start: list
    template_end: list

    def validate(self):
        assert len(self.query_end) \
           ==  len(self.template_end)  \
           ==  len(self.query_start)  \
           ==  len(self.template_start)  \
           ==  len(self.pdb_id)  \
           ==  len(self.chain_id) \
           ==  len(self.prob) \
           ==  len(self.p_value) \
           ==  len(self.e_value) \
           ==  len(self.cols) \
           ==  len(self.score)

    def select(self, start, stop):
        """
        Select all the lists from start to stop
        """
        self.pdb_id = self.pdb_id[start:stop]
        self.chain_id = self.chain_id[start:stop]
        self.description = self.description[start:stop]
        self.prob = self.prob[start:stop]
        self.e_value = self.e_value[start:stop]
        self.p_value = self.p_value[start:stop]
        self.score = self.score[start:stop]
        self.cols=self.cols[start:stop]
        self.query_start = self.query_start[start:stop]
        self.query_end = self.query_end[start:stop]
        self.template_start = self.template_start[start:stop]
        self.template_end = self.template_end[start:stop]

    def to_numeric(self):
        """
        Converts the string representation of the HHR hits to numeric representation 
        """

        self.prob = [float(j) for j in self.prob]
        self.e_value = [float(j) for j in self.e_value]
        self.p_value = [float(j) for j in self.p_value]
        self.score = [float(j) for j in self.score]

        for i in range(len(self.cols)):
            assert self.cols[i].strip(" ").isnumeric(), f"cols {self.cols[i]}"
            assert self.query_start[i].strip(" ").isnumeric()
            assert self.query_end[i].strip(" ").isnumeric()
            assert self.template_start[i].strip(" ").isnumeric()
            assert self.template_end[i].strip(" ").isnumeric()

        self.cols = [int(j) for j in self.cols]
        self.query_start = [int(j) for j in self.query_start]
        self.query_end = [int(j) for j in self.query_end]
        self.template_start = [int(j) for j in self.template_start]
        self.template_end = [int(j) for j in self.template_end]

RegExPiece = namedtuple("RegExPiece",
                        "no pdb_id chain_id prob e_value p_value score cols hmm") 



regex_piece = RegExPiece(no =      "^ *[0-9]{1,3}",
                         pdb_id =  "[A-Za-z0-9]{4}",
                         chain_id= "[A-Za-z0-9]+",
                         prob =    "[0-9]{1,3}\.[0-9]",
                         e_value = "[0-9\.E\-\+]{1,7}",
                         p_value = "[0-9\.E\-\+]{2,7}",
                         score =   "[0-9\.E\-]{3,6}",
                         cols =    "[0-9]{1,5}",
                         hmm =     "[0-9]{1,5}")

FirstFields = namedtuple("FirstFields",
                         "pdb_id chain_id")
SecondFields = namedtuple("SecondFields",
                          "prob e_value p_value score")
ThirdFields = namedtuple("ThirdFields",
                         "cols query_start query_end template_start template_end")
class RegEx:
    """
    The HHR_Re class defines the functionality for
    compiling and matching regular expressions for
    .hhr files. These are the log files from hhblits.
    They can be found in data/interim/cullin_e3_ligase/hhblits_out/
    They have the suffix .log instead of the suffix .hhr
    
    Class Attributes

      Attributes that begin with 

    """

    pdb_id_chain_id = re.compile((
        f"{regex_piece.no} ({regex_piece.pdb_id})_"
        f"({regex_piece.chain_id}) +"))
    prob_e_value_p_value_score = re.compile(".{34} +" + (
        f"({regex_piece.prob}) +"
        f"({regex_piece.e_value}) +"
        f"({regex_piece.p_value}) +"
        f"({regex_piece.score}) +"
        f"0\.0 +"))
    cols_qstart_qend_tstart_tend = re.compile(".{63} +0\.0 +" + (
        f"({regex_piece.cols}) +({regex_piece.hmm})-({regex_piece.hmm}) +"
        f"({regex_piece.hmm})-({regex_piece.hmm}) *\([0-9]+\)"))

    # Define the pieces  

class HHR:
    """
    Depricated dataclass. Use HHResults
    """
    def __init__(self,
                 uid: str, match_columns, 
                 no_of_seqs,
                 neff,
                 searched_hmms,
                 date,
                 command,
                 pdb_id,
                 description: str,
                 chain_id: list,
                 note: list, prob: list,
                 e_value: list, p_value: list,
                 score: list, cols: list,
                 query_start: list, query_end: list,
                 template_start: list, template_end: list):
        self.uid = uid
        self.match_columns = match_columns
        self.no_of_seqs = no_of_seqs
        self.neff = neff
        self.searched_hmm = searched_hmm
        self.date = date
        self.command = command
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.note = note
        self.prob = prob
        self.e_value = e_value
        self.p_value = p_value
        self.score = score
        self.cols = cols
        self.query_start = query_start
        self.query_end = query_end
        self.template_start = template_start
        self.template_end = template_end

def parse_log(handle, is_uniprot_accesion_id=None):
    """
    Parse an HHBlits .log file into a python object
    Assumes the query is based on a uniprot id
    """

    if not is_uniprot_accesion_id:
        is_uniprot_accesion_id = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")

    out = {"uid": None, 
           "description": None,
           "match_columns": None,
           "no_of_seqs": None,
           "neff": None,
           "searched_hmms": None,
           "date": None,
           "command": None,
           "pdb_id": [], "chain_id": [],
           "note": [], "prob": [],
           "e_value": [], "p_value": [],
           "score": [], "cols": [],
           "query_start": [], "query_end": [],
           "template_start": [], "template_end": []}

    with open(handle, 'r') as f:
        line1 = f.readline().strip("\n")
        
        q, header = line1.split("         ")
        sp, uid, description = header.split("|") 
        assert q=="Query"
        assert sp=="sp"
        assert is_uniprot_accesion_id.match(uid)
        
        out["uid"] = uid
        out["description"] = description

        line2 = f.readline().strip("\n")
        m, n = line2.split(" ")
        assert m=="Match_columns"
        assert n.isnumeric(), f"{n}"
        n = int(n)
        out["match_columns"] = n

        line3 = f.readline().strip("\n")
        a, s = line3.split("    ")
        assert a=="No_of_seqs", a
        out["no_of_seqs"] = s

        line4 = f.readline().strip("\n")
        a, n = line4.split(" "*10)
        assert a=="Neff", a
        assert n.isnumeric()
        out["neff"]=n

        line5 = f.readline().strip("\n")
        a, n = line5.split(" ")
        assert a=="Searched_HMMs", a
        assert n.isnumeric()
        n = int(n)
        out["searched_hmms"] = n

        line6 = f.readline().strip("\n")
        a, s = line6.split(" "*10)
        assert a=="Date"
        out["date"] = s

        line7 = f.readline().strip("\n")
        c, s = line7.split(" "*7)
        assert c=="Command"
        out["command"] = s

        line8 = f.readline()
        assert line8=="\n"

        line9 = f.readline().strip("\n")
        assert line9== " No Hit" + " "*29 + "Prob E-value P-value  Score" \
                + "    SS Cols Query HMM  Template HMM", line9
       
        n_matches = 0
        for line_num, hitline in enumerate(f):

            hitline = hitline.strip("\n")
            #assert '\n' not in hitline
            #assert m.match(hitline), (line_num, hitline)

            if RegEx.pdb_id_chain_id.match(hitline): 
                n_matches += 1
                print(hitline)
#                log_out_split  = log_output.split(hitline)

                _,  *first_fields, _ = RegEx.pdb_id_chain_id.split(hitline)
                _, *second_fields, _ = RegEx.prob_e_value_p_value_score.split(hitline)
                _, *third_fields, _ = RegEx.cols_qstart_qend_tstart_tend.split(hitline)

                first_fields = FirstFields(*first_fields)
                second_fields = SecondFields(*second_fields)
                third_fields = ThirdFields(*third_fields)

                # update the dictionary

                out["pdb_id"].append(first_fields.pdb_id)
                out["chain_id"].append(first_fields.chain_id)
                out["prob"].append(second_fields.prob)
                out["e_value"].append(second_fields.e_value)
                out["p_value"].append(second_fields.p_value)
                out["score"].append(second_fields.score)
                out["cols"].append(third_fields.cols)
                out["query_start"].append(third_fields.query_start)
                out["query_end"].append(third_fields.query_end)
                out["template_start"].append(third_fields.template_start)
                out["template_end"].append(third_fields.template_end)
        assert n_matches > 8, f"matches {n_matches}\nhandle {handle}"

        return out

@dataclass
class DevEnv:
    path: str

def _get_dev_env():
    path = "../../data/interim/cullin_e3_ligase/hhblits_out/A0FGR8_query_pdb70.log"
    return DevEnv(path=path)

def example1():
    path = "../../data/interim/cullin_e3_ligase/hhblits_out/A0FGR8_query_pdb70.log"
    out = parse_log(path)
    return HHResults(**out)

