import numpy as np
def half_sum(n):
    assert n % 2 == 0
    ncopies = np.log2(n) 
    ncopies = int(ncopies)
    rs = 0
    for i in range(ncopies):
        rs += 2**i
    rs = rs / n
    return rs

if __name__ == "__main__":
    import sys
    print(half_sum(int(sys.argv[1])))
