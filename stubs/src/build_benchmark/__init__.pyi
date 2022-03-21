from ..typdefs import FilePath as FilePath
from typing import Any

base_url: str
query_base: str
query_fpath: str
query: Any
request_path: Any
response: Any
resource: Any

def get_pdb_ids(resource): ...
def append_newline_chars_to_list_entries(mylist): ...
def write_pdb_ids_to_csv(resource) -> None: ...
