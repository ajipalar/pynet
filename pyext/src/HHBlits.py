import re
import pandas as pd
from dataclasses import dataclass, KW_ONLY

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
    q_start: list
    q_end: list
    t_start: list
    t_end: list

    def validate(self):
        assert len(self.q_end) \
           ==  len(self.t_end)  \
           ==  len(self.q_start)  \
           ==  len(self.t_start)  \
           ==  len(self.pdb_id)  \
           ==  len(self.chain_id) \
           ==  len(self.prob) \
           ==  len(self.p_value) \
           ==  len(self.e_value) \
           ==  len(self.cols) \
           ==  len(self.score)
 





class HHR:
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
                 q_start: list, q_end: list,
                 t_start: list, t_end: list):
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
        self.q_start = q_start
        self.q_end = q_end
        self.t_start = t_start
        self.t_end = t_end

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
           "q_start": [], "q_end": [],
           "t_start": [], "t_end": []}

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

        log_output = re.compile("^ *[0-9]+ ([A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9])" \
                + "_([A-Za-z0-9]+) (.{18-23}) +([0-9]+\.[0-9]+) +([0-9]+\.[0-9]+(E\-[0-9]+)*)" \
                + " +([0-9]+\.[0-9]+) + [0-9]+\.[0-9]+ +([0-9]+) +([0-9])+\-([0-9]+) +" \
                + "([0-9]+)\-([0-9]+) +\([0-9]+\) *")

        r_pdb =     re.compile("^ *[0-9]+ ([A-Za-z0-9]{4})")
        r_chain =   re.compile("^ *[0-9]+ [A-Za-z0-9]{4}_([A-Za-z0-9]+)")
        r_prob =    re.compile(".{34} +([0-9]{1,3}\.[0-9])")
        r_e_value = re.compile(".{40} ([0-9\.E\- ]{7}) +[0-9]")
        r_p_value = re.compile(".{48} ([0-9\.E\- ]{7})  [0-9\.E\- ]{6}")
        r_score   = re.compile(".{56} ([0-9\.E\- ]{6})")
        r_cols =    re.compile(".{69} ([0-9 ]{4})  [0-9]{1,4}-[0-9]{1,4}  ")
        r_qhmm =    re.compile(".{74} ([0-9 ]{1,4})-([0-9 ]{1,4}) +[0-9 ]{1,4}-[0-9]{1,4} *\(")
        r_Thmm =    re.compile(".{84} ([0-9 ]{1,5})-([0-9 ]{1,4}) *\(")

        for line_num, hitline in enumerate(f):

            hitline = hitline.strip("\n")
            #assert '\n' not in hitline
            #assert m.match(hitline), (line_num, hitline)
           
            if r_pdb.match(hitline):
                print(line_num, hitline)
                log_out_split  = log_output.split(hitline)
                _, pdb_id, _   =      r_pdb.split(hitline)
                _, chain_id, _ =    r_chain.split(hitline)
                _, prob, _     =     r_prob.split(hitline)
                _, e_value, _  =  r_e_value.split(hitline)
                _, p_value, _  =  r_p_value.split(hitline)
                _, score , _   =    r_score.split(hitline)
                _, cols  , _   =     r_cols.split(hitline) 
                _, q_start, q_end, _ = r_qhmm.split(hitline)
                _, t_start, t_end, _ = r_Thmm.split(hitline)

                #r_description = re.compile("^ *[0-9]+ [A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]_[A-Za-z0-9]+" \
                #+ " (.{" + f"{24 - n_chain_id_chars}" + "}) ")

                #_, description, _ = r_description.split(hitline)

                out["pdb_id"].append(pdb_id)
                out["chain_id"].append(chain_id)
                out["prob"].append(prob)
                out["e_value"].append(e_value)
                out["p_value"].append(p_value)
                out["score"].append(score)
                out["cols"].append(cols)
                out["q_start"].append(q_start)
                out["q_end"].append(q_end)
                out["t_start"].append(t_start)
                out["t_end"].append(t_end)

        return out

def dev():
    path = "../../data/interim/cullin_e3_ligase/hhblits_out/A0FGR8_query_pdb70.log"
    return path

def example1():
    path = "../../data/interim/cullin_e3_ligase/hhblits_out/A0FGR8_query_pdb70.log"
    out = parse_log(path)
    return HHResults(**out)

