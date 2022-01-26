
# Search RESTful API over http using json payload

import json
import requests
try:
    from IMP.pynet.typdefs import (
        FilePath
    )
except ModuleNotFoundError:
    from ..typdefs import (
        FilePath
    )

# Author Aji Palar
base_url = "https://search.rcsb.org/rcsbsearch/v1/query"  # py str
query_base = "?json="  # py str
query_fpath = "ecoli_query.json"  # str fpath
query = json.load(open(query_fpath, 'r'))  # py dict
query = json.dumps(query)  # json str

print(f"Querying the RCSB with {query_fpath.split('.')[0]}\nProceed? [y/N]")
print(query)
if input() == "y":

    request_path = base_url + query_base + query  # url endpoint str
    print(f'querying {request_path}')
    request_path = request_path.encode()  # url endpoint str utf-8
    response = requests.get(request_path)  # REST get request Response
    print(f'{response.status_code}')
    resource = response.json()  # Python dict json

    for key in resource: print(key)

def get_pdb_ids(resource):
    pdbids = []
    for i, entry in enumerate(resource['result_set']):
        pdbids.append(f'{entry["identifier"]}\n')
    return pdbids


def append_newline_chars_to_list_entries(mylist):
    ammended_list = []
    for item in mylist:
        item = f'{item}\n'
        ammended_list.append(item)
    return ammended_list


def write_pdb_ids_to_csv(resource):
    with open("response.csv", 'w') as f:
        f.writelines(get_pdb_ids(resource))


write_pdb_ids_to_csv(resource)
