def pipe_parse(s: str)-> str:
    """
    x | f | g -> g(f(x))
    Inputs should be pure functions
    """

    assert type(s) == str
    s=s.replace(" ", "")
    s=s.replace("\t", "")
    l_s: list[str] = s.split('|')
    assert len(l_s) > 0
    output = ""
    for arg in l_s:
        output = f'({arg}{output})'
    return output[1:-1]


def pipe(fluid, *segments):
    """
    The parameters flow through the pipe from input to output
    Only use pure segments, don't use impure dirty piping
    Those lead to oil spills
    """
    for segment in segments:
        fluid = segment(fluid)
    return fluid


def pipe_depracated(s: str):
    """Evaluates the pipe s and returns the value"""
    pipes = pipe_parse(s)
    return eval(pipes)
