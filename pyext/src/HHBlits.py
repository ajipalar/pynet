import re
import pandas as pd
class Log:
    def __init__(self,
                 uid: str, description: str,
                 hit: list, chain: list,
                 note: list, prob: list,
                 e_value: list, p_value: list,
                 score: list, cols: list,
                 query_hmm: list, template_hmm: list):
        self.hit = hit
        self.chain = chain
        self.note = note
        self.prob = prob
        self.e_value = e_value
        self.p_value = p_value
        self.score = score
        self.cols = cols
        self.query_hmm = query_hmm
        self.template_hmm = template_hmm

def parse_log(handle, is_uniprot_accesion_id=None):
    """
    Parse an HHBlits .log file into a python object
    Assumes the query is based on a uniprot id
    """

    if not is_uniprot_accesion_id:
        is_uniprot_accesion_id = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")

    out = {"uid": None, 
           "description": None,
           "Match_columns": None,
           "No_of_seqs": None,
           "Neff": None,
           "Searched_HMMs": None,
           "Date": None,
           "Command": None,
           "hit": [], "chain": [],
           "note": [], "prob": [],
           "e_value": [], "p_value": [],
           "score": [], "cols": [],
           "query_hmm": [], "template_hmm": []}

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
        out["Match_columns"] = n

        line3 = f.readline().strip("\n")
        a, s = line3.split("    ")
        assert a=="No_of_seqs", a
        out["No_of_seqs"] = s

        line4 = f.readline().strip("\n")
        a, n = line4.split(" "*10)
        assert a=="Neff", a
        assert n.isnumeric()
        out["Neff"]=n

        line5 = f.readline().strip("\n")
        a, n = line5.split(" ")
        assert a=="Searched_HMMs", a
        assert n.isnumeric()
        n = int(n)
        out["Searched_HMMs"] = n

        line6 = f.readline().strip("\n")
        a, s = line6.split(" "*10)
        assert a=="Date"
        out["Date"] = s

        line7 = f.readline().strip("\n")
        c, s = line7.split(" "*7)
        assert c=="Command"
        out["Command"] = s

        line8 = f.readline()
        assert line8=="\n"

        line9 = f.readline().strip("\n")
        assert line9== " No Hit" + " "*29 + "Prob E-value P-value  Score" \
                + "    SS Cols Query HMM  Template HMM", line9

        
def parse_hit_line(hitline):
    """

    """
    ...









def dev():
    path = "../../data/interim/cullin_e3_ligase/hhblits_out/A0FGR8_query_pdb70.log"
    return path

def example1():
    path = "../../data/interim/cullin_e3_ligase/hhblits_out/A0FGR8_query_pdb70.log"
    parse_log(path)


        
