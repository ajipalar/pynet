def exhaust(it):
    for i in it: pass

def forp(it):
    """Prints the contents of the iterator, consuming it"""
    exhaust(map(print, it))
