#From Functional Programing HOWTO 0.32 A. M. Kuchling

#Object orientated -> little state capsules
#Functional -> Data flow

L = [1, 2, 3]
it = iter(L)
print(it)
it.__next__()

#Python sequence types support iteration and iterators
#str, list, dictionary (over keys), Sets
#py > 3.7 iteration order same as insertion order
#Files support iterations with readline


def my_generator(x):
    i=0
    while i < x:
        val = (yield i)
        if val is not None:
            i = val
        else:
            i +=1

def my_gen2(x):
    yield x


def my_gen3(x):
    while True:
        val = (yield x)
        print(val)

#Generators can be created by generator functions
#Generators can hold state information
#Information can be sent to genertors
#Generators process information sequentially
#Generators have next, send, throw, and close methods
#Generators do not resume after StopIteration
#throw raises exceptions
#sub routines entrance -> exit
#co routines enter -> exit -> pause -> enter


def counter(maximum):
    i = 0
    while i < maximum:
        val = (yield i)
        if val is not None:
            i = val
        else:
            i += 1
