def modparse(local_dict):
    modname = local_dict['__name__']
    if "." in modname:
        modname = modname.split(".")[1]
    packname = local_dict['__package__']
    return modname, packname
def moduleinfo(local_dict):
    modname, packname = modparse(local_dict)
    print(f'[Package, Module]: {packname} {modname}')

def doc(x):
    """
    Print the docstring
    """
    print(x.__doc__)

import config
if config.PRINT_MODULE_INFO:
    moduleinfo(locals())
