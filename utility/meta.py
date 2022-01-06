def pipe_parse(s: str)-> str:
    """
    x | f | g -> g(f(x))
    Inputs should be pure functions 
    """

    assert type(s) == str
    s=s.replace(" ", "")
    s=s.replace("\t", "")
    l = s.split('|')
    assert len(l) > 0
    output = ""
    for arg in l:
        output = f'({arg}{output})'
    return output[1:-1]

def pipe(*args):
    for i in args:
        print(i)

def pipe(s: str):
    """Evaluates the pipe s and returns the value"""
    pipes = pipe_parse(s)
    return eval(pipes)

