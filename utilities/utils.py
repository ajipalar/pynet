def modparse(local_dict):
    modname = local_dict['__name__']
    modname = modname.split(".")[0]
    packname = local_dict['__package__']
    return modname, packname
def moduleinfo(modname, packname):
    print(f'[Package, Module]: {packname} {modname}')

import config
if config.PRINT_MODULE_INFO:
    moduleinfo(locals())
